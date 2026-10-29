# Independent Biot ADER-DG solvers

Copy this entire `dg/` folder to use the wave solvers and dataset generators
independently. No neural-network model, training script, checkpoint or installed
neural-operator package is required. PyTorch is used for DG tensor calculations.
This is a standalone script collection, not a pip-installable library.

**All commands on this page and the model pages assume your terminal is inside
this `dg/` folder.** In the complete repository, first run `cd dg`.

## Choose a workflow

| Workflow | Solver and helpers | Entry point | Hardware |
|---|---|---|---|
| [Two-layer](two_layer/README.md) | `two_layer/two_layer_coarse_gpu_colab.py` | Same file for a single reference simulation; `two_layer/generate_biot_picno_dataset_gpu.py` for randomized HDF5 data | CUDA required by these entrypoints |
| [Faulted three-layer](faulted_three_layer/README.md) | `faulted_three_layer/aderdg_core.py`, `fault_geometry.py`, `materials.py`, `interface_flux.py` | `faulted_three_layer/generate_faulted_biot_dataset.py` | CPU or CUDA |

Keep each model folder's Python files together. Imports and source-hash checks
resolve sibling files. There are no imports from training folders. CPU NumPy
(and Triangle for the faulted geometry) builds meshes; PyTorch executes DG tensor
operations on the selected device. GPU-oriented does not mean GPU meshing.

## Installation

Python 3.12 was used for source checks. Create an environment and install a
PyTorch build appropriate for your CPU/CUDA hardware. The PyTorch installation
selector is at https://pytorch.org/get-started/locally/ . Then:

```bash
python -m venv .venv
source .venv/bin/activate
# Install your selected PyTorch build in this environment, then:
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
mkdir -p data results
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The commands below use shell line continuation as written for Linux/macOS bash.
Dependencies are not pinned to a frozen numerical environment. Record package
versions and hardware for experiments. CPU PyTorch supports the faulted checks;
it does not make the two-layer CUDA-only entrypoints CPU-compatible.

## First inspect the faulted geometry

```bash
python faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/geometry_preview.h5 --preview-only
```

This creates `data/geometry_preview_figures/geometry_mesh.png`; it does not
simulate waves or create `geometry_preview.h5`. The included
[faulted geometry image](faulted_three_layer/geometry_mesh_preview.png) is a
previously packaged preview, not a new validation result.

## Small CPU execution example

```bash
python faulted_three_layer/check_faulted_model.py
python faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_cpu_smoke_1_1_1.h5 \
  --train 1 --validation 1 --test 1 --device cpu \
  --degree 1 --max-area 4000 --nx 16 --ny 16 \
  --final-time 0.008 --frame-interval 0.002 --skip-plots
python faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_cpu_smoke_1_1_1.h5 --verify-only
```

These intentionally coarse settings exercise execution and storage; they are
not scientifically resolved labels, and were not executed during reorganization.
The 0.008 s run ends before the Ricker peak at 0.04 s. It checks the workflow,
not the full pulse or its transmission. Use a new output path for another mesh,
degree, duration, split size or numerical configuration.

## Run one two-layer reference simulation

```bash
python two_layer/two_layer_coarse_gpu_colab.py
```

This CUDA-only entrypoint runs the fixed nominal material/acquisition setup
and saves an NPZ output, a solver checkpoint and plots under
`coarse_gpu_results/` relative to the current directory. It does not train a
network or generate train/validation/test splits. Configuration is through the
constants at the top of the file, not command-line options. See the
[two-layer model page](two_layer/README.md) for outputs and limitations.

## Generate the full datasets

Two-layer:

```bash
python -u two_layer/generate_biot_picno_dataset_gpu.py \
  --train 3000 --validation 200 --test 200 --skip-paper \
  --frame-interval 0.002 --final-time 0.5 \
  --output data/biot_picno_coarse_3000_200_200.h5 \
  --figure-dir results/two_layer_generation
```

Faulted three-layer:

```bash
python -u faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_biot_3000_200_200.h5 \
  --train 3000 --validation 200 --test 200 --device cuda
```

Both preserve an 8/2/2 pilot default; counts above are explicit. For an ordinary
pilot, use 8/2/2 and a different output filename. These numbers count independent
simulated realizations. They do not count DG time steps, training epochs or
optimizer updates. The two files describe different geological models.

At the default saved grid and times, each full dataset stores wavefields with
shape `[3400 total realizations, 251 times, 8 states, 120 y, 128 x]` across its
three split groups. Raw float32 wavefields alone total about **391 GiB**, before
compression and other arrays. The DG mesh is separate from this sampling grid.
Start with a pilot and inspect storage capacity and numerical behavior.

To resume, rerun the exact generation command with unchanged numerical files and
settings. Completed realizations are skipped; incomplete ones restart at time
zero. This is distinct from the single two-layer solver's step checkpoint.
Verify stored completed fields with:

```bash
python two_layer/generate_biot_picno_dataset_gpu.py \
  --output data/biot_picno_coarse_3000_200_200.h5 --verify-only
python faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_biot_3000_200_200.h5 --verify-only
```

Completion markers and configuration hashes do not guarantee recovery from a
damaged HDF5 file. Do not read for training while the file is being written.
The `paper` label in two-layer options means nominal published parameters on
this project's mesh, not an independently reproduced paper solution.

## Governing equations and discretization

The state is `U = [sxx, syy, sxy, vx, vy, p, qx, qy]`, with

    U_t + A U_x + B U_y = E U + S.

The bundled simulations are isotropic and inviscid: viscous drag E is zero.
Stress/pressure are Pa; solid velocity and relative fluid flux q are m/s,
where `q = porosity * (fluid_velocity - solid_velocity)`. Permeability is recorded
but does not actively control viscous attenuation in these datasets.

Modal polynomials live on triangular DG elements. The default degree is 4
(15 modes per state per element). A local Taylor ADER predictor forms integrated
states; volume derivatives and neighboring face fluxes update each element.
Same-material faces use Rusanov-type fluxes. The faulted workflow replaces
material-changing face fluxes with a bonded/open-pore characteristic solve.
Zero exterior traces provide approximate absorption, not a PML.

Nominal O5 does not establish fifth-order convergence for this forced,
heterogeneous geometry. In particular, the predictor contains no stiff viscous
treatment, and source addition follows predictor/flux integration. Quantitative
source, interface and convergence benchmarks remain necessary.

## Customization and external use

| Change | Where to look | Consequence |
|---|---|---|
| Two-layer mesh, source, receivers, degree, material constants | Top of `two_layer/two_layer_coarse_gpu_colab.py` | Use a fresh checkpoint/output; generator source hashes change |
| Two-layer split sizes and saved times | Generator `--help` | Use explicit counts; new settings need a new HDF5 file |
| Faulted geometry, fault tips/dips/throws, acquisition | `faulted_three_layer/fault_geometry.py` | Rebuild mesh; use a new dataset |
| Faulted material sampling and mixing | `faulted_three_layer/materials.py` | Record assumptions and admissibility checks |
| Faulted resolution, precision, device, saved times | Generator `--help` | CPU smoke choices differ from production defaults |
| DG basis, predictor and update | Each core's `ModalBasis` and `GPUADERDG` classes | Numerical-method change requiring separate verification |

The faulted core must be configured with the intended materials/acquisition and
have `interface_flux.install` applied before heterogeneous stepping. The generator
shows this sequence. Importing `aderdg_core.py` alone does not reproduce the
complete faulted setup. The existing entrypoints are the documented public path;
these internal classes are not a stabilized general simulation API.

## Attribution, validation and license

Published methods are credited to [de la Puente et al. (2008), Geophysics](https://doi.org/10.1190/1.2965027)
and the [de la Puente thesis](https://edoc.ub.uni-muenchen.de/8074/), especially
Section 4.7.4 and Table 4.6 for the two-layer benchmark. The original theory is
not a new contribution of this repository. This is a project implementation,
with AI coding assistance reported in its development documentation, not the
paper authors' original source release or a verified exact reproduction.

The two-layer mesh is coarser than the published benchmark; the source coefficient
2e10 is a project assumption. Fault geometry, effective middle-layer mixing,
material perturbations, dataset sampling and GPU implementation are project
choices. No scientific novelty, measured speedup or production accuracy is
established here. Faults are static bonded/open-pore offsets, not dynamic rupture.

Retained solver/generator Python bytes are unchanged from the supplied archive.
Syntax and static dependency checks were performed after relocation; simulations,
training and numerical checks were not rerun. Historical checks reported in the
model documentation are not independently reproduced results from this pass.
A complete validation needs homogeneous solutions, mesh/time convergence,
interface benchmarks, boundary-reflection measurements and recorded environments.

No redistribution license has been selected. See [LICENSE_STATUS.md](LICENSE_STATUS.md).
Technical independence of this folder does not establish redistribution rights.
Maintainer: Fawz Naim. Cite the software snapshot and original methods when used.
