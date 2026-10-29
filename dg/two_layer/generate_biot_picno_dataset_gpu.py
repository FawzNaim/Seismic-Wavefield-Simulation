"""Full-state inviscid Biot dataset for supervised and physics-informed CNO.

Run with CUDA on Linux. Keep two_layer_coarse_gpu_colab.py beside this file.
Derived from the user's earlier coarse-generator builder, not a newly verified
replication of de la Puente's thesis. The solver is unchanged numerically.
Outputs [sxx, syy, sxy, vx, vy, p, qx, qy]; y corresponds to vertical z.
Defaults: 8/2/2 pilot, 128x120 output grid, O5/17280 triangles,
0 <= t <= 0.5 s, frame spacing 0.002 s (251 frames).
Matrices A/B/E and sampled projected source are included for residual checks.
Rerun the exact same command to skip completed realizations. A stopped
incomplete realization is rerun from time zero, not resumed at its last step.
This generator does not implement a physics-informed loss or certify residual
accuracy. Reference residual calibration must precede physics training.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# USER CONTROLS
# =============================================================================

# Requested large-dataset defaults.
DEFAULT_TRAIN_SAMPLES = 8
DEFAULT_VALIDATION_SAMPLES = 2
DEFAULT_TEST_SAMPLES = 2

DATASET_FILE = Path("biot_picno_pilot_8_2_2.h5")
FIGURE_DIR = Path("biot_picno_dataset_figures")

# Uniform grid seen by the FNO.  It is independent of the triangular DG mesh.
OUTPUT_NX = 128
OUTPUT_NY = 120

# 0, 0.01, ..., 0.50 s.  The 50 Hz source period is 0.02 s, so this gives two
# saved frames per source period.  Set 0.005 for denser temporal supervision.
FRAME_INTERVAL_S = 0.002
FINAL_TIME_S = 0.50

# State ordering in the ADER-DG solver:
# [sxx, syy, sxy, vx, vy, p, qx, qy]
OUTPUT_COMPONENTS = tuple(range(8))
OUTPUT_NAMES = ("sxx", "syy", "sxy", "vx", "vy", "p", "qx", "qy")
OUTPUT_UNITS = ("Pa", "Pa", "Pa", "m/s", "m/s", "Pa", "m/s", "m/s")

RANDOM_SEED = 20260925

# Independent, bounded perturbations about the paper values.  The exact paper
# values are never drawn or inserted into the random splits.
SOLID_DENSITY_FRACTION = 0.05
MODULUS_FRACTION = 0.10
POROSITY_ABSOLUTE = 0.02
TORTUOSITY_FRACTION = 0.10
FLUID_DENSITY_FRACTION = 0.02
FLUID_BULK_FRACTION = 0.10

# Compression level 1 is a useful compromise between disk size and Colab CPU
# time.  Use None if disk is plentiful and writing becomes the bottleneck.
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_LEVEL = 1

# Preserve the paper dt for the exact-paper holdout.  For randomized models,
# dt is reduced automatically if the perturbed fastest wave speed requires it.
RANDOM_DT_SAFETY = 0.90

INPUT_NAMES = (
    "rho_s",
    "rho_f",
    "lambda_matrix",
    "mu_matrix",
    "bulk_solid",
    "bulk_fluid",
    "porosity",
    "log10_permeability",
    "tortuosity",
    "material_id",
)

PARAMETER_NAMES = tuple(
    f"{layer}_{name}"
    for layer in ("shale", "sandstone")
    for name in INPUT_NAMES[:-1]
)


# =============================================================================
# LOAD THE COARSE GPU SOLVER
# =============================================================================

def load_solver_module():
    """Find the previously supplied solver in Colab or beside this file."""
    candidates = [
        Path(__file__).with_name("two_layer_coarse_gpu_colab.py"),
        Path.cwd() / "two_layer_coarse_gpu_colab.py",
        Path("/content/two_layer_coarse_gpu_colab.py"),
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("coarse_gpu_solver", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    searched = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "Upload two_layer_coarse_gpu_colab.py before running this generator. "
        f"Searched:\n{searched}"
    )


# =============================================================================
# MATERIAL SAMPLING
# =============================================================================

def varied(value, fraction, rng):
    return float(value * rng.uniform(1.0 - fraction, 1.0 + fraction))


def valid_material(material):
    """Reject unphysical draws before launching a costly simulation."""
    try:
        if not 0.01 < material.porosity < 0.60:
            return False
        if material.bulk_matrix >= material.bulk_solid:
            return False
        if material.mass_determinant <= 0.0:
            return False
        if material.coupling_M <= 0.0:
            return False
        speeds = material.wave_speeds()
        return len(speeds) > 0 and np.all(np.isfinite(speeds))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return False


def sample_materials(base, sample_index, split_code):
    """Return a deterministic, physically admissible two-layer realization."""
    seed = np.random.SeedSequence([RANDOM_SEED, split_code, sample_index])
    rng = np.random.default_rng(seed)

    # Brine properties are global, so both rocks contain the same fluid.
    rho_f = varied(base.SHALE.rho_f, FLUID_DENSITY_FRACTION, rng)
    bulk_fluid = varied(base.SHALE.bulk_fluid, FLUID_BULK_FRACTION, rng)

    def draw(reference):
        return base.BiotMaterial(
            name=reference.name,
            rho_s=varied(reference.rho_s, SOLID_DENSITY_FRACTION, rng),
            rho_f=rho_f,
            lambda_matrix=varied(
                reference.lambda_matrix, MODULUS_FRACTION, rng
            ),
            mu_matrix=varied(reference.mu_matrix, MODULUS_FRACTION, rng),
            bulk_solid=varied(reference.bulk_solid, MODULUS_FRACTION, rng),
            bulk_fluid=bulk_fluid,
            porosity=float(
                reference.porosity
                + rng.uniform(-POROSITY_ABSOLUTE, POROSITY_ABSOLUTE)
            ),
            permeability=reference.permeability,
            tortuosity=varied(
                reference.tortuosity, TORTUOSITY_FRACTION, rng
            ),
            viscosity=0.0,
        )

    # Rejection sampling is deterministic because the RNG stream is fixed.
    for _ in range(1000):
        shale = draw(base.SHALE)
        sandstone = draw(base.SANDSTONE)
        if valid_material(shale) and valid_material(sandstone):
            return shale, sandstone
    raise RuntimeError("Could not draw physically admissible materials")


def material_vector(materials):
    values = []
    for material in materials:
        values.extend(
            [
                material.rho_s,
                material.rho_f,
                material.lambda_matrix,
                material.mu_matrix,
                material.bulk_solid,
                material.bulk_fluid,
                material.porosity,
                np.log10(material.permeability),
                material.tortuosity,
            ]
        )
    return np.asarray(values, dtype=np.float64)


def make_input_fields(base, x, y, materials):
    """Create piecewise-constant physical-property maps on the FNO grid."""
    _, Y = np.meshgrid(x, y)
    fields = np.empty((len(INPUT_NAMES), len(y), len(x)), dtype=np.float32)
    masks = (Y >= base.INTERFACE_Y_M, Y < base.INTERFACE_Y_M)
    for mask, material, material_id in zip(masks, materials, (0.0, 1.0)):
        values = (
            material.rho_s,
            material.rho_f,
            material.lambda_matrix,
            material.mu_matrix,
            material.bulk_solid,
            material.bulk_fluid,
            material.porosity,
            np.log10(material.permeability),
            material.tortuosity,
            material_id,
        )
        for channel, value in enumerate(values):
            fields[channel, mask] = value
    return fields


def make_source_map(base, x, y):
    """A narrow visualization/input map; the DG label still uses a point source."""
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    sigma = 1.5 * max(dx, dy)
    source = np.exp(
        -0.5
        * (
            ((X - base.SOURCE_X_M) / sigma) ** 2
            + ((Y - base.SOURCE_Y_M) / sigma) ** 2
        )
    )
    return source.astype(np.float32)


# =============================================================================
# TIME INTEGRATION AND SAMPLING
# =============================================================================

def frame_times():
    if FRAME_INTERVAL_S <= 0 or FINAL_TIME_S <= 0:
        raise ValueError("Frame interval and final time must be positive")
    count = int(round(FINAL_TIME_S / FRAME_INTERVAL_S))
    if count < 4 or not np.isclose(count * FRAME_INTERVAL_S, FINAL_TIME_S,
                                 rtol=0, atol=1e-12):
        raise ValueError("Final time must be an integer multiple of frame interval, with >=5 frames")
    return np.linspace(0.0, FINAL_TIME_S, count + 1, dtype=np.float64)


def build_schedule(base, requested_times, nominal_dt):
    """Use small DG steps but end a step exactly at every stored FNO frame."""
    requested = list(requested_times[1:])
    steps, end_times, source_integrals = [], [], []
    t = 0.0
    frame = 0
    eps = 20.0 * np.finfo(float).eps
    while t < requested_times[-1] - eps:
        dt = min(nominal_dt, requested_times[-1] - t)
        if frame < len(requested) and t < requested[frame] < t + dt:
            dt = requested[frame] - t
        steps.append(dt)
        source_integrals.append(base.integrated_ricker(t, dt))
        t += dt
        end_times.append(t)
        if frame < len(requested) and abs(t - requested[frame]) < 1.0e-10:
            frame += 1
    return (
        np.asarray(steps, dtype=np.float64),
        np.asarray(end_times, dtype=np.float64),
        np.asarray(source_integrals, dtype=np.float64),
    )


@torch.no_grad()
def simulate_one(
    base,
    mesh,
    device,
    x,
    y,
    requested_times,
    materials,
    paper_case=False,
    keep_receivers=False,
):
    """Run one ADER-DG realization and sample it on the common FNO grid."""
    # GPUADERDG reads these module globals during construction.
    base.SHALE, base.SANDSTONE = materials
    base.MATERIALS = tuple(materials)

    solver = base.GPUADERDG(
        mesh, base.POLYNOMIAL_DEGREE, device, base.TORCH_DTYPE
    )
    X, Y = np.meshgrid(x, y)
    output_points = np.column_stack((X.ravel(), Y.ravel()))
    output_sampler = solver.make_sampler(output_points)
    receiver_sampler = solver.make_sampler(base.RECEIVERS_M)
    projected_source = np.moveaxis(
        output_sampler.sample(solver.source).reshape(len(y), len(x), 8), -1, 0
    ).astype(np.float32)
    matrices = [material.matrices() for material in materials]
    matrix_A = np.stack([pair[0] for pair in matrices])
    matrix_B = np.stack([pair[1] for pair in matrices])

    if paper_case:
        nominal_dt = base.PAPER_DT_S
    else:
        nominal_dt = (
            RANDOM_DT_SAFETY * float(solver.cfl_diagnostic_s)
        )
    dt_steps, end_times, source_integrals = build_schedule(
        base, requested_times, nominal_dt
    )

    coefficients = torch.zeros(
        (len(mesh.elements), solver.nmodes, solver.nvars),
        device=device,
        dtype=base.TORCH_DTYPE,
    )
    outputs = np.zeros(
        (
            len(requested_times),
            len(OUTPUT_COMPONENTS),
            len(y),
            len(x),
        ),
        dtype=np.float32,
    )
    receiver_times = [0.0]
    receiver_history = [
        receiver_sampler.sample(coefficients)[:, OUTPUT_COMPONENTS]
    ]

    next_frame = 1
    started = time.perf_counter()
    report_every = max(1, len(dt_steps) // 20)
    for step, (dt, end_time, source_integral) in enumerate(
        zip(dt_steps, end_times, source_integrals), start=1
    ):
        coefficients = solver.step(
            coefficients, float(dt), float(source_integral)
        )
        if keep_receivers:
            receiver_times.append(float(end_time))
            receiver_history.append(
                receiver_sampler.sample_gpu(coefficients)[:, OUTPUT_COMPONENTS]
                .cpu()
                .numpy()
            )

        if (
            next_frame < len(requested_times)
            and abs(end_time - requested_times[next_frame]) < 1.0e-9
        ):
            sampled = output_sampler.sample(coefficients)
            sampled = sampled.reshape(len(y), len(x), solver.nvars)
            outputs[next_frame] = np.moveaxis(
                sampled[:, :, OUTPUT_COMPONENTS], -1, 0
            )
            if not np.isfinite(outputs[next_frame]).all():
                raise FloatingPointError(f"Nonfinite wavefield at t={end_time}")
            next_frame += 1

        if step % report_every == 0 or step == len(dt_steps):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (len(dt_steps) - step)
            maximum = float(torch.max(torch.abs(coefficients)).item())
            print(
                f"    step {step:,}/{len(dt_steps):,}, "
                f"t={end_time:.4f} s, |Q|max={maximum:.3e}, "
                f"ETA={eta / 60.0:.1f} min"
            )

    if next_frame != len(requested_times):
        raise RuntimeError(
            f"Stored {next_frame}/{len(requested_times)} requested frames"
        )

    diagnostics = {
        "nominal_dt_s": nominal_dt,
        "number_of_steps": len(dt_steps),
        "cfl_diagnostic_s": float(solver.cfl_diagnostic_s),
        "runtime_s": time.perf_counter() - started,
        "matrix_A": matrix_A,
        "matrix_B": matrix_B,
        "projected_source": projected_source,
    }
    del coefficients, output_sampler, receiver_sampler, solver
    gc.collect()
    torch.cuda.empty_cache()

    if keep_receivers:
        return (
            outputs,
            diagnostics,
            np.asarray(receiver_times, dtype=np.float64),
            np.asarray(receiver_history, dtype=np.float32),
        )
    return outputs, diagnostics, None, None


# =============================================================================
# RESUMABLE HDF5 FILE
# =============================================================================

def string_array(values):
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def create_or_validate_file(path, counts, x, y, times, base):
    exists = path.exists()
    handle = h5py.File(path, "a")

    configuration = {
        "nx": len(x),
        "ny": len(y),
        "times_s": times.tolist(),
        "output_components": list(OUTPUT_COMPONENTS),
        "output_names": list(OUTPUT_NAMES),
        "input_names": list(INPUT_NAMES),
        "random_seed": RANDOM_SEED,
        "counts": counts,
        "polynomial_degree": base.POLYNOMIAL_DEGREE,
        "triangles": int(2 * base.NX * base.NY),
        "schema_version": "biot_picno_full_state_v1",
        "domain_x_m": list(base.DOMAIN_X_M),
        "domain_y_m": list(base.DOMAIN_Y_M),
        "interface_y_m": base.INTERFACE_Y_M,
        "source": [base.SOURCE_X_M, base.SOURCE_Y_M,
                   base.SOURCE_FREQUENCY_HZ, base.SOURCE_DELAY_S,
                   base.POINT_SOURCE_STRENGTH],
        "solver_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    encoded = json.dumps(configuration, sort_keys=True)
    if exists:
        previous = handle.attrs.get("configuration_json", "")
        if previous != encoded:
            handle.close()
            raise ValueError(
                "Existing dataset has a different configuration. Choose a new "
                "--output filename or restore the original settings."
            )
        return handle

    handle.attrs["description"] = (
        "Two-layer inviscid Biot ADER-DG labels for neural-operator training"
    )
    handle.attrs["configuration_json"] = encoded
    handle.attrs["paper_case_excluded_from_random_splits"] = True
    handle.attrs["labels_are_analytical"] = False
    handle.attrs["labels_are_numerical_ader_dg"] = True

    coords = handle.create_group("coords")
    coords.create_dataset("x_m", data=x)
    coords.create_dataset("y_m", data=y)
    coords.create_dataset("times_s", data=times)

    metadata = handle.create_group("metadata")
    metadata.create_dataset("input_names", data=string_array(INPUT_NAMES))
    metadata.create_dataset("output_names", data=string_array(OUTPUT_NAMES))
    metadata.create_dataset("output_units", data=string_array(OUTPUT_UNITS))
    metadata.create_dataset("matrix_layer_names", data=string_array(("shale", "sandstone")))
    metadata.attrs["state_equation"] = "U_t + A U_x + B U_y = E U + S"
    metadata.attrs["stress_convention"] = "total stress, tension positive; pore pressure compression positive"
    metadata.attrs["fluid_variable"] = "q = porosity * (fluid_velocity - solid_velocity)"
    metadata.attrs["boundary_condition"] = "solver exterior trace zero in upwind numerical flux"
    metadata.attrs["source_map_role"] = "Gaussian input/visualization map, NOT the DG source"
    metadata.attrs["projected_source_role"] = "sampled modal DG Dirac projection; not an exact continuum delta"
    metadata.attrs["source_temporal_function"] = "Ricker; see bundled solver ricker/integrated_ricker"
    metadata.attrs["matrix_layer_id_mapping"] = "0 shale; 1 sandstone"
    metadata.attrs["frame_interval_s"] = float(times[1] - times[0])
    metadata.attrs["initial_condition"] = "all eight fields zero"
    metadata.create_dataset(
        "parameter_names", data=string_array(PARAMETER_NAMES)
    )
    metadata.create_dataset(
        "receiver_locations_m", data=base.RECEIVERS_M.astype(np.float64)
    )
    metadata.attrs["source_x_m"] = base.SOURCE_X_M
    metadata.attrs["source_y_m"] = base.SOURCE_Y_M
    metadata.attrs["source_frequency_hz"] = base.SOURCE_FREQUENCY_HZ
    metadata.attrs["source_delay_s"] = base.SOURCE_DELAY_S
    metadata.attrs["source_strength"] = base.POINT_SOURCE_STRENGTH
    metadata.attrs["interface_y_m"] = base.INTERFACE_Y_M
    metadata.attrs["upper_layer"] = "shale"
    metadata.attrs["lower_layer"] = "sandstone"
    metadata.attrs["viscosity_pa_s"] = 0.0

    static = handle.create_group("static")
    static.create_dataset("source_map", data=make_source_map(base, x, y))

    compression_options = (
        HDF5_COMPRESSION_LEVEL if HDF5_COMPRESSION == "gzip" else None
    )
    for split, count in counts.items():
        group = handle.create_group(split)
        group.create_dataset(
            "inputs",
            shape=(count, len(INPUT_NAMES), len(y), len(x)),
            dtype="f4",
            chunks=(1, len(INPUT_NAMES), len(y), len(x)),
            compression=HDF5_COMPRESSION,
            compression_opts=compression_options,
        )
        group.create_dataset(
            "wavefields",
            shape=(
                count,
                len(times),
                len(OUTPUT_NAMES),
                len(y),
                len(x),
            ),
            dtype="f4",
            chunks=(1, 1, len(OUTPUT_NAMES), len(y), len(x)),
            compression=HDF5_COMPRESSION,
            compression_opts=compression_options,
        )
        group.create_dataset(
            "parameters", shape=(count, len(PARAMETER_NAMES)), dtype="f8"
        )
        for matrix_name in ("matrix_A", "matrix_B", "matrix_E"):
            group.create_dataset(matrix_name, shape=(count, 2, 8, 8), dtype="f8")
        group.create_dataset("completed", data=np.zeros(count, dtype=np.uint8))
        group.create_dataset("runtime_s", data=np.full(count, np.nan))
        group.create_dataset("nominal_dt_s", data=np.full(count, np.nan))
    handle.flush()
    return handle


def save_physics_fields(handle, group, index, diagnostics):
    for name in ("matrix_A", "matrix_B"):
        if index is None:
            group.create_dataset(name, data=diagnostics[name])
        else:
            group[name][index] = diagnostics[name]
    zeros = np.zeros((2, 8, 8), dtype=np.float64)
    if index is None:
        group.create_dataset("matrix_E", data=zeros)
    else:
        group["matrix_E"][index] = zeros
    static = handle["static"]
    source = diagnostics["projected_source"]
    if "projected_source" not in static:
        static.create_dataset("projected_source", data=source, compression="gzip")
    elif not np.array_equal(static["projected_source"][:], source):
        raise ValueError("Shared DG source changed between realizations")


def save_random_sample(group, index, inputs, outputs, parameters, diagnostics):
    if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
        raise FloatingPointError("Cannot commit nonfinite simulation")
    group["completed"][index] = 0
    group["inputs"][index] = inputs
    group["wavefields"][index] = outputs
    group["parameters"][index] = parameters
    group["runtime_s"][index] = diagnostics["runtime_s"]
    group["nominal_dt_s"][index] = diagnostics["nominal_dt_s"]
    save_physics_fields(group.file, group, index, diagnostics)
    group.file.flush()
    group["completed"][index] = 1
    group.file.flush()


def save_paper_reference(
    handle,
    inputs,
    outputs,
    materials,
    diagnostics,
    receiver_times,
    receiver_history,
):
    if "paper_reference" in handle:
        del handle["paper_reference"]
    group = handle.create_group("paper_reference")
    group.create_dataset("inputs", data=inputs, compression=HDF5_COMPRESSION)
    group.create_dataset(
        "wavefields",
        data=outputs,
        chunks=(1, len(OUTPUT_NAMES), outputs.shape[-2], outputs.shape[-1]),
        compression=HDF5_COMPRESSION,
        compression_opts=(
            HDF5_COMPRESSION_LEVEL if HDF5_COMPRESSION == "gzip" else None
        ),
    )
    group.create_dataset("parameters", data=material_vector(materials))
    group.create_dataset("receiver_times_s", data=receiver_times)
    group.create_dataset("receiver_history", data=receiver_history)
    save_physics_fields(handle, group, None, diagnostics)
    for key, value in diagnostics.items():
        if key not in ("matrix_A", "matrix_B", "projected_source"):
            group.attrs[key] = value
    handle.flush()
    group.attrs["completed"] = True
    handle.flush()


# =============================================================================
# VISUAL CHECKS
# =============================================================================

def robust_limit(field):
    return max(float(np.percentile(np.abs(field), 99.5)), 1.0e-30)


def plot_paper_holdout(handle, base, figure_dir):
    """Create direct visual checks corresponding to paper Figures 9 and 10."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = handle["coords/x_m"][:]
    y = handle["coords/y_m"][:]
    times = handle["coords/times_s"][:]
    group = handle["paper_reference"]
    wavefields = group["wavefields"]
    inputs = group["inputs"]

    vy_channel = OUTPUT_NAMES.index("vy")
    time_index = int(np.argmin(np.abs(times - 0.25)))
    vy = wavefields[time_index, vy_channel]
    limit = robust_limit(vy)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), constrained_layout=True)
    material = axes[0].pcolormesh(
        x, y, inputs[INPUT_NAMES.index("material_id")], shading="auto",
        cmap="cividis"
    )
    axes[0].axhline(base.INTERFACE_Y_M, color="white", ls="--", lw=1.5)
    axes[0].scatter(base.SOURCE_X_M, base.SOURCE_Y_M, marker="*", s=160,
                    color="red", edgecolor="white")
    axes[0].scatter(base.RECEIVERS_M[:, 0], base.RECEIVERS_M[:, 1], marker="v",
                    s=50, color="black")
    axes[0].set_title("Held-out paper configuration")
    fig.colorbar(material, ax=axes[0], label="0 shale, 1 sandstone")

    image = axes[1].pcolormesh(
        x, y, vy, shading="auto", cmap="seismic", vmin=-limit, vmax=limit
    )
    axes[1].axhline(base.INTERFACE_Y_M, color="black", ls="--", lw=0.8)
    axes[1].set_title(f"Paper holdout: $v_y$ at t={times[time_index]:.2f} s")
    fig.colorbar(image, ax=axes[1], label="$v_y$ (m/s)")
    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
    snapshot_path = figure_dir / "paper_holdout_figure9_check.png"
    fig.savefig(snapshot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    receiver_times = group["receiver_times_s"][:]
    receiver_history = group["receiver_history"][:]
    vx_channel = OUTPUT_NAMES.index("vx")
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True,
                             constrained_layout=True)
    for receiver in range(3):
        for column, (channel, label) in enumerate(
            ((vx_channel, "$v_x$"), (vy_channel, "$v_y$"))
        ):
            trace = receiver_history[:, receiver, channel]
            scale = max(np.max(np.abs(trace)), 1.0e-30)
            axes[receiver, column].plot(receiver_times, trace / scale, lw=1.0)
            axes[receiver, column].grid(alpha=0.25)
            axes[receiver, column].set_ylabel(
                f"R{receiver + 1} {label}\nnormalized"
            )
    axes[0, 0].set_title("Horizontal solid velocity")
    axes[0, 1].set_title("Vertical solid velocity")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Held-out paper case: normalized receiver traces")
    receiver_path = figure_dir / "paper_holdout_figure10_check.png"
    fig.savefig(receiver_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {snapshot_path}")
    print(f"Saved {receiver_path}")


def plot_first_training_example(handle, figure_dir):
    if len(handle["train/completed"]) == 0 or not handle["train/completed"][0]:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = handle["coords/x_m"][:]
    y = handle["coords/y_m"][:]
    times = handle["coords/times_s"][:]
    time_index = int(np.argmin(np.abs(times - 0.25)))
    pressure = handle["train/wavefields"][0, time_index, OUTPUT_NAMES.index("p")]
    porosity = handle["train/inputs"][0, INPUT_NAMES.index("porosity")]
    limit = robust_limit(pressure)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    a = axes[0].pcolormesh(x, y, porosity, shading="auto", cmap="viridis")
    fig.colorbar(a, ax=axes[0], label="porosity")
    b = axes[1].pcolormesh(
        x, y, pressure / 1.0e6, shading="auto", cmap="seismic",
        vmin=-limit / 1.0e6, vmax=limit / 1.0e6
    )
    fig.colorbar(b, ax=axes[1], label="p (MPa)")
    axes[0].set_title("First training input")
    axes[1].set_title(f"First training label at t={times[time_index]:.2f} s")
    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
    path = figure_dir / "first_training_example.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# =============================================================================
# MAIN DATASET DRIVER
# =============================================================================


def generate_dataset(output_path, counts, paper_only=False, skip_paper=False):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Colab choose Runtime > "
            "Change runtime type > GPU."
        )

    if paper_only and skip_paper:
        raise ValueError(
            "--paper-only and --skip-paper cannot be used together"
        )

    base = load_solver_module()

    # Preserve the published shale and sandstone objects. simulate_one changes
    # the solver module globals for each randomized realization.
    exact_shale = base.SHALE
    exact_sandstone = base.SANDSTONE
    exact_materials = (exact_shale, exact_sandstone)

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    x = np.linspace(
        base.DOMAIN_X_M[0],
        base.DOMAIN_X_M[1],
        OUTPUT_NX,
    )
    y = np.linspace(
        base.DOMAIN_Y_M[0],
        base.DOMAIN_Y_M[1],
        OUTPUT_NY,
    )
    times = frame_times()

    expected_triangles = 2 * base.NX * base.NY

    print("Building the shared coarse mesh once...")
    mesh = base.build_paper_mesh()

    if len(mesh.elements) != expected_triangles:
        raise RuntimeError(
            f"Expected {expected_triangles:,} triangles, "
            f"created {len(mesh.elements):,}"
        )

    print(f"  cells       : {base.NX} x {base.NY}")
    print(f"  triangles   : {len(mesh.elements):,}")
    print(f"  output grid : {OUTPUT_NX} x {OUTPUT_NY}")
    print(f"  saved frames: {len(times)}")
    print(f"  receivers   : {len(base.RECEIVERS_M)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize or validate the single large HDF5 file, then close it.
    # The file stays closed during every expensive ADER-DG simulation.
    initializer = create_or_validate_file(
        output_path,
        counts,
        x,
        y,
        times,
        base,
    )
    initializer.flush()
    initializer.close()

    split_codes = {
        "train": 11,
        "validation": 22,
        "test": 33,
    }

    if not paper_only:
        for split in ("train", "validation", "test"):
            for index in range(counts[split]):

                # Open the HDF5 file only long enough to inspect completion.
                with h5py.File(output_path, "r") as check:
                    completed = bool(
                        check[f"{split}/completed"][index]
                    )

                if completed:
                    print(
                        f"Skipping completed {split} "
                        f"sample {index + 1}/{counts[split]}"
                    )
                    continue

                # Reset the published reference materials before each
                # deterministic randomized draw.
                base.SHALE, base.SANDSTONE = exact_materials

                materials = sample_materials(
                    base,
                    index,
                    split_codes[split],
                )

                print(
                    f"\nRunning coarse {split} sample "
                    f"{index + 1}/{counts[split]}"
                )

                # The large HDF5 file is CLOSED during this simulation.
                outputs, diagnostics, _, _ = simulate_one(
                    base,
                    mesh,
                    device,
                    x,
                    y,
                    times,
                    materials,
                )

                inputs = make_input_fields(
                    base,
                    x,
                    y,
                    materials,
                )

                parameters = material_vector(materials)

                # Briefly reopen the file and commit exactly one complete
                # realization. The completion flag is written last.
                with h5py.File(output_path, "r+") as commit:
                    save_random_sample(
                        commit[split],
                        index,
                        inputs,
                        outputs,
                        parameters,
                        diagnostics,
                    )
                    commit.flush()

                print(
                    f"Completed {split} sample "
                    f"{index + 1}/{counts[split]}; "
                    f"runtime="
                    f"{diagnostics['runtime_s'] / 60.0:.2f} min; "
                    f"dt={diagnostics['nominal_dt_s']:.6e} s"
                )

                # Release CPU copies before beginning the next simulation.
                del outputs, inputs, parameters
                gc.collect()

    # Determine whether a paper-property reference is already stored.
    with h5py.File(output_path, "r") as check:
        paper_complete = (
            "paper_reference" in check
            and bool(
                check["paper_reference"].attrs.get(
                    "completed",
                    False,
                )
            )
        )

    if not skip_paper and not paper_complete:
        print("\nRunning the separate paper-property holdout...")

        base.SHALE, base.SANDSTONE = exact_materials

        # This is the paper material/source configuration evaluated using
        # the coarse mesh. The original fine-paper solver remains untouched.
        outputs, diagnostics, receiver_times, receiver_history = simulate_one(
            base,
            mesh,
            device,
            x,
            y,
            times,
            exact_materials,
            paper_case=True,
            keep_receivers=True,
        )

        inputs = make_input_fields(
            base,
            x,
            y,
            exact_materials,
        )

        with h5py.File(output_path, "r+") as commit:
            save_paper_reference(
                commit,
                inputs,
                outputs,
                exact_materials,
                diagnostics,
                receiver_times,
                receiver_history,
            )
            commit.flush()

        del outputs, inputs, receiver_times, receiver_history
        gc.collect()

    elif paper_complete:
        print("Skipping completed paper-property holdout")

    # Final verification and summary.
    with h5py.File(output_path, "r") as handle:
        print("\nDataset summary")

        for split in ("train", "validation", "test"):
            completed = int(
                np.sum(handle[f"{split}/completed"][:])
            )
            total = len(handle[f"{split}/completed"])

            print(
                f"  {split:10s}: "
                f"{completed}/{total} complete"
            )

        print(
            "  paper case : "
            + (
                "complete"
                if "paper_reference" in handle
                else "missing"
            )
        )

        if (
            len(handle["train/completed"]) > 0
            and bool(handle["train/completed"][0])
        ):
            plot_first_training_example(
                handle,
                FIGURE_DIR,
            )

        if "paper_reference" in handle:
            plot_paper_holdout(
                handle,
                base,
                FIGURE_DIR,
            )

    size_gb = output_path.stat().st_size / (1024**3)

    print(f"  file       : {output_path.resolve()}")
    print(f"  file size  : {size_gb:.2f} GiB")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate full-state coarse Biot data for physics-informed CNO"
    )
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--validation", type=int, default=DEFAULT_VALIDATION_SAMPLES
    )
    parser.add_argument("--test", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--output", type=Path, default=DATASET_FILE)
    parser.add_argument("--frame-interval", type=float, default=FRAME_INTERVAL_S)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME_S)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--verify-only", action="store_true",
                        help="check all stored completed fields without running simulations")
    parser.add_argument(
        "--paper-only", action="store_true",
        help="run only the exact-paper holdout (split sizes still define the file)"
    )
    parser.add_argument(
        "--skip-paper", action="store_true",
        help="generate random splits without running the exact-paper holdout"
    )
    return parser.parse_args()


def verify_full_state_dataset(path):
    with h5py.File(path, "r") as f:
        names = tuple(v.decode() if isinstance(v, bytes) else str(v)
                      for v in f["metadata/output_names"][:])
        if names != OUTPUT_NAMES:
            raise ValueError(f"Expected full-state ordering {OUTPUT_NAMES}; found {names}")
        times = f["coords/times_s"][:]
        if len(times) < 5 or not np.all(np.diff(times) > 0) or not np.allclose(
                np.diff(times), times[1]-times[0], rtol=1e-8, atol=1e-12):
            raise ValueError("Nonuniform or insufficient saved times")
        nx, ny = len(f["coords/x_m"]), len(f["coords/y_m"])
        checked = 0
        for split in ("train", "validation", "test"):
            g = f[split]
            completed = g["completed"][:]
            if g["wavefields"].shape != (len(completed), len(times), 8, ny, nx):
                raise ValueError(f"Incorrect wavefield shape in {split}")
            for index in np.flatnonzero(completed):
                if not np.isfinite(g["inputs"][index]).all():
                    raise ValueError(f"Nonfinite material map: {split}/{index}")
                for name in ("matrix_A", "matrix_B", "matrix_E"):
                    if g[name].shape != (len(completed), 2, 8, 8) or not np.isfinite(g[name][index]).all():
                        raise ValueError(f"Invalid {name}: {split}/{index}")
                for frame in range(len(times)):
                    if not np.isfinite(g["wavefields"][index, frame]).all():
                        raise ValueError(f"Nonfinite field: {split}/{index}/{frame}")
                if np.any(g["wavefields"][index, 0] != 0):
                    raise ValueError("Initial state differs from zero initial condition")
                checked += 1
            print(f"{split}: {int(completed.sum())}/{len(completed)} complete")
        if checked:
            source = f["static/projected_source"][:]
            if source.shape != (8, ny, nx) or not np.isfinite(source).all():
                raise ValueError("Invalid projected source")
        if "paper_reference" in f and f["paper_reference"].attrs.get("completed", False):
            g = f["paper_reference"]
            if g["wavefields"].shape != (len(times), 8, ny, nx):
                raise ValueError("Invalid reference shape")
            for frame in range(len(times)):
                if not np.isfinite(g["wavefields"][frame]).all():
                    raise ValueError("Nonfinite reference field")
        print(f"Verified {checked} completed random realizations, {len(times)} frames, eight fields")
        print("This verifies storage integrity; physics residual accuracy is a separate check.")


def main():
    global FRAME_INTERVAL_S, FINAL_TIME_S, FIGURE_DIR
    args = parse_arguments()
    if args.verify_only:
        verify_full_state_dataset(args.output.expanduser())
        return
    FRAME_INTERVAL_S, FINAL_TIME_S = args.frame_interval, args.final_time
    FIGURE_DIR = args.figure_dir
    times = frame_times()
    counts = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Use positive train/validation/test counts; use --paper-only for reference only")
    gib = sum(counts.values()) * len(times) * 8 * OUTPUT_NY * OUTPUT_NX * 4 / 1024**3
    print(f"Uncompressed wavefield payload: {gib:.2f} GiB (compression depends on data)")
    generate_dataset(
        args.output.expanduser(),
        counts,
        paper_only=args.paper_only,
        skip_paper=args.skip_paper,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# GOOGLE COLAB COMMANDS
# =============================================================================
#
# 1. Select Runtime > Change runtime type > GPU.
# 2. Upload both Python files.
# 3. Run a small, resumable pilot:
#
#    !python generate_biot_picno_dataset_gpu.py \
#        --train 8 --validation 2 --test 2
#
# 4. After checking the figures, create a larger dataset under a NEW filename:
#
#    !python generate_biot_picno_dataset_gpu.py \
#        --train 64 --validation 12 --test 12 \
#        --output shale_sandstone_fno_dataset_64_12_12.h5
#
# Re-running exactly the same command resumes at the first incomplete sample.
