# Two-layer Biot DG workflow

Independent CUDA/PyTorch solver and randomized eight-state dataset generator.
Keep both Python files together. All commands here assume the terminal is in
`dg/`; see [installation and common instructions](../README.md).

## Files and model

- `two_layer_coarse_gpu_colab.py`: fixed nominal reference simulation, DG classes,
  mesh creation, material matrices, source, point sampling and plotting.
- `generate_biot_picno_dataset_gpu.py`: randomized material realizations, uniform
  sampling, full-state HDF5 storage, resume and verification; no network training.

The domain is x=0..1500 m and y=0..1400 m, with a shale/sandstone interface at
y=700 m. The source is at (750,900) m; receivers at (950,750), (950,650), (950,500) m.
The Ricker source has 50 Hz peak frequency and delay 0.04 s, with an assumed
coefficient 2e10 acting with opposite signs on syy and p. Coordinates use solver
y upward. The dataset's 120×128 sampling grid is not the DG mesh.

The coarse mesh uses 96×90 cells split into 17,280 triangles; degree 4 supplies
15 modal coefficients per state/element. The implementation uses a local Taylor
ADER predictor and Rusanov-type face fluxes. Its earlier two-layer face treatment
is not the faulted workflow's characteristic material-interface solve. Exterior
zero traces are approximately absorbing; no PML or sponge is supplied.

This uses nominal parameters from de la Puente's heterogeneous benchmark, not
its fine 132,014-triangle mesh. Historical code comments claiming a configuration
match or an “unreleased mesh” do not establish exact reproduction or source-code
unavailability. Nominal O5 and an exact scalar source integral are not proof of
global convergence for this source/interface configuration.

## Run a single reference simulation

```bash
python two_layer/two_layer_coarse_gpu_colab.py
```

CUDA is required by this entrypoint. Edit configuration constants at the top of
the script for a different setup; there is no argparse CLI. Output is relative
to your current working directory:

| Output | Contents |
|---|---|
| `coarse_gpu_results/coarse_gpu_two_layer_output.npz` | Sampled simulation outputs |
| `coarse_gpu_results/coarse_gpu_checkpoint.pt` | DG solver step checkpoint |
| `coarse_gpu_results/*.png` | Model, field and receiver plots |

It automatically resumes an existing step checkpoint when configured to do so.
Use a new `OUTPUT_DIR` or disable checkpoint reuse when changing numerical
settings. This checkpoint is unrelated to neural-network weights. The single
run's NPZ is not the generator's eight-state training HDF5 schema.

## Generate independent realizations

Pilot:

```bash
python -u two_layer/generate_biot_picno_dataset_gpu.py \
  --train 8 --validation 2 --test 2 --skip-paper \
  --output data/biot_picno_pilot_8_2_2.h5 \
  --figure-dir results/two_layer_pilot
```

Full dataset:

```bash
python -u two_layer/generate_biot_picno_dataset_gpu.py \
  --train 3000 --validation 200 --test 200 --skip-paper \
  --frame-interval 0.002 --final-time 0.5 \
  --output data/biot_picno_coarse_3000_200_200.h5 \
  --figure-dir results/two_layer_generation
```

Counts are realizations, not iterations. Defaults remain 8/2/2; the documented
full dataset is 3000/200/200. This generator requires CUDA and does not expose a
`--device` option. Both files stay unchanged numerically. The word `paper` in CLI
options identifies nominal published properties on this coarse project mesh.

Rerun the same command to resume completed-realization storage. Changing split
counts or settings requires a different filename. Verify without generating:

```bash
python two_layer/generate_biot_picno_dataset_gpu.py \
  --output data/biot_picno_coarse_3000_200_200.h5 --verify-only
```

## Contents

Saved frame spacing is 0.002 s: 251 frames at t=0,...,0.5 s, ten samples per
50 Hz period. This is finer than the old 0.01 s spacing; adequacy for PDE
residuals must still be measured. Use --frame-interval 0.001 for 501 frames
under a new output name if needed. This does not change the source frequency
or nominal stable ADER-DG timestep (steps land exactly at saved frame times).

State channels are exactly:
[sxx, syy, sxy, vx, vy, p, qx, qy].
The second axis is named y by the solver, corresponding to vertical z.
Stress and pressure are in Pa. Solid velocities and q are in m/s;
q = porosity * (fluid_velocity - solid_velocity), a relative volume flux.

HDF5 layout:
coords/{x_m,y_m,times_s}
metadata/{input_names,output_names,output_units,parameter_names,matrix_layer_names,...}
static/source_map — Gaussian visualization/input map, NOT numerical point source
static/projected_source — sampled spatial DG source projection including strength
train|validation|test/inputs — [N,10,120,128]
train|validation|test/wavefields — [N,251,8,120,128]
train|validation|test/{matrix_A,matrix_B,matrix_E} — [N,2,8,8]
train|validation|test/{parameters,completed,runtime_s,nominal_dt_s}

A and B use U_t + A U_x + B U_y = E U + S; E is zero because viscosity is
zero. Layer IDs: 0 shale, 1 sandstone. Source projection is multiplied by the
solver's Ricker time function; it is a sampled DG projection, not an exact
continuum Dirac distribution. Metadata records source location, amplitude,
frequency, delay, interface, conventions, and solver/generator hashes.

The same deterministic parameter sampling as the earlier coarse generator
is retained. Saved time sampling differs, so trajectories are not necessarily
bit-for-bit identical to the old dataset. For controlled architecture comparisons,
train both supervised and physics-informed models on this new dataset with the
same three-field evaluation channels and the same train/validation/test split.

Without --skip-paper the paper-property reference is saved separately on the
coarse mesh; it is not the publication's exact fine mesh. To add it later, rerun
with the SAME counts/output/time settings and omit --skip-paper. Random samples
already completed will be skipped.

## Storage and validation

Raw float32 wavefields for 3000/200/200 at 251 frames occupy approximately
391 GiB before compression, metadata and additional arrays. The 8/2/2 pilot is
about 1.38 GiB for wavefields alone. Gzip is enabled; actual sizes vary.

Verification checks completion, shapes and finite values, not numerical accuracy.
No CUDA simulation was rerun during folder reorganization. Validate homogeneous
wave speeds, mesh/time convergence, interfaces and exterior reflections before
using these fields as quantitatively verified labels. The source routine and
material-interface treatment particularly merit independent benchmarks.

For the original methods and license status, see [DG documentation](../README.md).
