# Full-state Biot physics-informed CNO trainer

Keep train_biot_picno.py, cno_model.py, and biot_physics.py together.
Requires CUDA-enabled PyTorch, numpy, h5py, matplotlib in your existing environment.
The generator and solver files are not needed for training. Dataset is opened read-only.

## Setup on your Linux workstation

Extract this folder into /path/to/biot-neural-operators/biot_picno_training.
Activate your existing aderdg environment. In the Linux SSH terminal:

    TRAINER=/path/to/biot-neural-operators/biot_picno_training/train_biot_picno.py
    PICNO_DATA=/actual/verified/path/biot_picno_coarse_3000_200_200.h5
    PICNO_OUT=/path/to/biot-neural-operators/picno_results

Replace PICNO_DATA with the SAME HDF5 path used for your successful verification.
Find it if needed:

    find /path/to/data -type f -name 'biot_picno_coarse_3000_200_200.h5'

These variables belong to the current shell; reassign them after opening a new shell.
Run long jobs in tmux to keep them running after SSH disconnection.

## 1. Reference residual calibration (recommended first)

    python -u "$TRAINER" --data "$PICNO_DATA" --output-dir "$PICNO_OUT" \
      --history-frames 4 --batch-size 2 --calibration-pairs 64 --calibrate-only

The first run calculates normalization by streaming ONLY the training split.
This reads the training trajectories once and may take time; progress is printed.
normalization.json is reused for the same dataset and normalization stride.
--statistics-frame-stride 10 speeds this up by using every tenth training frame;
use the same stride for supervised and physics-informed comparisons.
Defaults use all frames, not subsampling.

Inspect reference_residual_calibration.json and .png plus physics_mask.png.
The bars are dimensionless residual RMS, NOT prediction errors in percent.
Reference residuals are not forced to zero or subtracted from predictions.
If reference residuals are large or highly variable, examine temporal/spatial
sampling and derivative consistency before assigning a strong physics weight.
A small residual is necessary evidence, not a proof of correct numerical physics.
The CLI does not automatically decide whether the calibration is acceptable.

## 2. Physics-informed training

    python -u "$TRAINER" --data "$PICNO_DATA" --output-dir "$PICNO_OUT" \
      --history-frames 4 --epochs 150 --no-early-stopping \
      --batch-size 2 --width 32 --layers 2 --windows-per-realization 16 \
      --physics-loss-weight 0.01 --plot-times 0.1,0.2,0.3,0.4,0.5

0.01 is an initial trial value, not an established optimal weight. Change it only
after inspecting reference residuals and the supervised-versus-physics losses.
The script repeats a sampled calibration at startup, then starts training.
Optional --physics-warmup-epochs 10 ramps only the physics weight; it does not
introduce recursive prediction or change the history length.

## 3. Controlled supervised comparison on this SAME dataset

    python -u "$TRAINER" --data "$PICNO_DATA" \
      --output-dir /path/to/biot-neural-operators/cno_full_state_baseline \
      --history-frames 4 --epochs 150 --no-early-stopping \
      --batch-size 2 --width 32 --layers 2 --windows-per-realization 16 \
      --physics-loss-weight 0 --plot-times 0.1,0.2,0.3,0.4,0.5

Also keep optional interface/boundary weights at zero for a purely supervised run.
Compare with this baseline, rather than assuming comparability with the old
three-output model: new history duration, time spacing, and outputs differ.
Both modes also report the unlearned 2*current-previous extrapolation baseline.

## Mathematical implementation and scope

State order: [sxx,syy,sxy,vx,vy,p,qx,qy], y corresponds to vertical z.
Four TRUE full states (32 field channels), ten material maps, source map and
normalized target time give 44 input channels. Two coordinates are added inside
CNO. Next frame has eight output channels. Prediction is a learned correction
to 2*U_n-U_(n-1) in normalized coordinates; all four frames enter the network.
There is NO predicted feedback and NO recursive rollout.

CNO architecture: spatial convolutions, residual feature blocks, antialiased
bicubic resizing, oversampled LeakyReLU and encoder/decoder skips. This is a
portable approximation, not the original CNO's custom sinc-filter implementation.
Width and down/up levels remain configurable; changed architecture requires
new training. Existing three-output checkpoints cannot be loaded here.

Residual in physical units:
R = (3*Uhat_(n+1)-4*Utrue_n+Utrue_(n-1))/(2*dt_saved)
    + A*d_x(Uhat_(n+1)) + B*d_y(Uhat_(n+1)) - E*Uhat_(n+1) - S(t_(n+1)).
S uses static/projected_source times the recorded Ricker time function.
Spatial derivatives use fourth-order centered five-point stencils and spacings
from coords/x_m,y_m. Matrices A,B,E are read from each realization, selected by
material ID per cell. E is zero in the supplied inviscid dataset. This trainer
explicitly rejects nonzero-viscosity datasets to avoid implying support for a
solver extension not included in the provided generator.

Normalize each equation by sigma_c/Tstar using TRAINING field standard deviations
and --physics-time-scale (0.02 s by default). Biot loss is mean squared scaled
residual over fields, batch, and valid cells. PDE computation is float32 outside
network mixed precision; metrics accumulate in float64.

Bulk stencils avoid the exterior (4 cells), material interfaces (3 cells around
a change), and the sampled DG source support (4 cells) by default. All fields
are supervised on the entire grid, including these regions. This is a correct
bulk Biot residual, not the acoustic approximation, but default training does
NOT separately enforce interface or absorbing-boundary equations. Finite-
difference residuals on resampled DG fields are not identical to the DG scheme.

Two optional additions are implemented but OFF by default:
--interface-loss-weight: continuity of vx,vy,p,qy,syy,sxy across the horizontal
interface, using linear one-sided trace extrapolation. Assumes a perfectly
bonded, open-pore interface. sxx and qx are not incorrectly forced continuous.
--boundary-loss-weight: zero incoming characteristic flux, using the negative
part of the outward-normal Biot matrix in normalized coordinates, divided by
maximum wave speed. This is approximate absorption, not a perfectly absorbing
boundary and not exact replication of a finite-resolution DG numerical flux.

To CALIBRATE both additions before using them:

    python -u "$TRAINER" --data "$PICNO_DATA" --output-dir "$PICNO_OUT" \
      --calibrate-only --interface-loss-weight 0.001 --boundary-loss-weight 0.001

Their reference penalties are included in the calibration JSON when enabled.
Use them for training only if their assumptions and measured reference penalties
are acceptable. The default initial trial isolates bulk PDE regularization.

Total loss:
L = wavefield_balanced + 0.1*gradient_matching + 0.1*RMS_amplitude_matching
    + lambda_PDE*Biot + lambda_interface*interface + lambda_boundary*boundary.
RMS matching is not physical Biot energy conservation.

## Training, evaluation, and checkpoints

--windows-per-realization 16 samples up to 16 target windows per TRAIN realization
per epoch, changing the sample each epoch. 0 uses all available training windows.
Validation uses ALL windows by default; --validation-windows 16 uses a fixed
subset if full validation is too costly. Test evaluation always uses ALL windows.
No training/validation/test split leakage; normalization and calibration use train.

Best checkpoint is selected by validation channel-balanced relative L2 across
p,vx,vy; all eight per-field relative L2 values are also reported. This preserves
the main scientific outputs for comparison. It is not plateau detection.
--no-early-stopping runs all requested epochs. --evaluate-final selects the
final checkpoint instead of the best checkpoint after training.

Outputs:
- normalization.json
- reference_residual_calibration.json/.png and physics_mask.png
- best_picno.pt and final_picno.pt (with optimizer/scheduler/scaler for resume)
- training_history.json/.png (supervised, physics, validation error curves)
- test_one_step_metrics.json (eight fields, main metric, extrapolation baseline)
- test_one_step_error_vs_time.png and test_error_vs_time.npz
- selected snapshots for realization zero: p,vx,vy plots; all eight arrays in NPZ

Initial four history frames are excluded from prediction metrics and snapshot
requests. Saved snapshots show nearest saved time; these are one-step predictions
using true history. Figures show physical x/y coordinates and units.

## Resume training

Repeat the training command with unchanged architecture, loss, history and sampling
settings, add:

    --resume "$PICNO_OUT/final_picno.pt"

--epochs is the desired TOTAL epoch count, not additional epochs. Resume restores
optimizer/scheduler/scaler and previous history; it resumes between completed
epochs. Prefer the same output directory, especially for best/final selection.

## Evaluate another snapshot without retraining

    python -u "$TRAINER" --data "$PICNO_DATA" --output-dir "$PICNO_OUT" \
      --evaluate-only --checkpoint "$PICNO_OUT/best_picno.pt" \
      --history-frames 4 --plot-times 0.15,0.35,0.45

Network width/layers are loaded from the checkpoint in evaluation-only mode.
Dataset configuration and history/input shape are checked against the checkpoint.
Only load checkpoints you trust; PyTorch loading here includes training metadata.

## Validation performed before delivery

Analytic manufactured solutions tested time/spatial derivatives, matrix signs,
zero residual for an exact linear PDE solution, nonzero differentiable physics
penalty after perturbation, and interface continuity. Small CPU synthetic-data
smoke runs tested training with PDE/interface/boundary terms, checkpoint saving,
evaluation-only equivalence and resumed training. These tests validate software,
not scientific accuracy on your actual dataset. Full CUDA training and your
reference residual calibration remain to be run on the Linux workstation.

## Dataset generation

Use [the separate DG package](../dg/README.md) to generate data. The full example
uses 3000 training, 200 validation and 200 test realizations. These counts are
not training epochs. Supply the actual HDF5 path; no DG solver code is required
inside this training folder. Smaller pilots also work if schema-compatible.
