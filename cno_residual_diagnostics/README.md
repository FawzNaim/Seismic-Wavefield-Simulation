# Post-training residual diagnostics for the eight-field history CNO

Evaluate your EXISTING 1.3%-error checkpoint and the original
`biot_picno_coarse_3000_200_200.h5`. No training or dataset regeneration.
This tool is for the original four-true-history, one-step mapping. It also
accepts the spectral-history variant. Direct hard-IC and three-field CNO
checkpoints are intentionally rejected.

Extract this folder into `/path/to/biot-neural-operators/`, activate `aderdg`, and
keep all included Python files together. Use CUDA PyTorch, NumPy, h5py and
matplotlib from your existing environment. Supply the actual weight-zero
run's checkpoint path; do not accidentally select the higher-error PICNO run.

```bash
python -u /path/to/biot-neural-operators/cno_residual_diagnostics/evaluate_cno_residuals.py \
  --data /path/to/biot-neural-operators/biot_picno_coarse_3000_200_200.h5 \
  --checkpoint /ACTUAL/PATH/TO/YOUR_ZERO_PHYSICS_RUN/best_picno.pt \
  --output-dir /path/to/biot-neural-operators/cno_zero_physics_residuals \
  --split test --mode both \
  --batch-size 8 --residual-batch-size 8 \
  --residual-dtype float64 \
  --plot-times 0.1,0.2,0.3,0.4,0.5
```

Default realization count is zero, meaning ALL 200 test realizations. For
a quick pilot add `--realizations 2` and use a separate output directory.
No DataLoader workers are used: this streams one realization at a time,
reading its fields once, then performs batched inference and diagnostics.
Only test fields are used for the reported test errors. The stored training
normalization and architecture are restored from the checkpoint. No new
statistics computation or optimizer update is performed. The H5 is read-only.

## Equations

State order is sxx, syy, sxy, vx, vy, p, qx, qy. Here y corresponds to vertical
z. The q variables are relative filtration velocities/volume flux per unit
area, q=porosity*(fluid velocity-solid velocity), measured in m/s; they are
not stored fluid volumes. Residuals are the EIGHT COUPLED equations, not
eight independent scalar acoustic wave equations:

    R = (3 U_next - 4 U_previous + U_older)/(2 dt)
        + A D_x^(4) U_next + B D_z^(4) U_next - E U_next - S(t_next)

dt is the saved .002-s interval, not the ADER-DG internal integration step.
A/B/E are the saved matrices for each realization and layer; E is zero for
the current inviscid dataset. S is the actual sampled projected DG source
times the Ricker wavelet, not the Gaussian network source descriptor.

This is the same BDF2/fourth-order bulk residual formula as the earlier PICNO
regularizer, evaluated in float64 by default to reduce arithmetic cancellation.
It is NOT the original ADER-DG discrete update. Predictions and labels still
have their original finite precision. `--residual-dtype float32` is available
for a precision sensitivity check. Model inference uses the saved AMP setting
unless `--no-amp` is supplied; changing AMP can slightly change predictions.

## Two complementary checks

1. `teacher_forced`: predicted U_next, TRUE previous/older states. This is
   the residual the old physics loss penalized. With history=4 and 251 frames,
   all 247 predicted targets per realization are eligible (49,400 total).
2. `predicted_sequence`: predicted U_next, predicted U_previous and predicted
   U_older. Each of those states is independently generated with its own
   TRUE four-frame history. This tests whether the assembled prediction
   sequence has internally consistent bulk dynamics. It is NOT an autonomous
   recursive rollout. Only times with all THREE states predicted are included:
   index >=6, 245 targets per realization (49,000 total). No copied seed frames
   are used in this check.

ADER-DG reference residuals use true states at all three times, evaluated on
EXACTLY the same target times/cells as each corresponding CNO check. Thus the
reference summaries can differ slightly between modes because their eligible
time sets differ. `--mode teacher` or `--mode sequence` runs one check only.

The source/interface/exterior mask is restored from checkpoint settings,
defaulting to margins 4/3/4. Full fourth-order stencils must stay inside a
material and away from the source projection. Optional `--boundary-cells`,
`--interface-cells`, `--source-cells` override these margins (minimum 2).
The tool reports and plots the actual mask. These tests do NOT certify source
injection, boundary conditions, initial conditions, or interface jump laws.

## What is reported

- Physical prediction L2 over ALL original eligible one-step frames and
  full spatial grid. It should approximately reproduce the existing test
  metric; minor numerical differences can arise from inference/batch/precision.
- Physical residual RMS: Pa/s for stress and pressure equations; m/s^2 for
  solid-velocity and filtration-velocity equations. Different-unit equations
  must not be judged by their raw RMS magnitudes against each other.
- Dimensionless residual: R * T_star / training_std(channel), with T_star
  restored from the checkpoint (normally .02 s). Override using
  `--physics-time-scale` only for an explicitly labelled scaling experiment.
- Dimensionless RMS, mean absolute, maximum absolute, and signed mean for
  both CNO and reference. The signed mean alone can cancel to zero even for
  large residuals, which is why RMS/absolute measures are provided.
- RMS of R_CNO-R_reference, CNO/reference RMS ratio and difference/reference
  RMS ratio. These are relative RESIDUAL comparisons, not wavefield error.
- `relative_balance_rms`: RMS(R) divided by the SUM of the RMS values of
  Ut, A Ux, B Uz, E U and S on the same samples/cells. This dimensionless
  closure measure is generally in [0,1]. It is not a prediction percentage
  or a universal acceptance criterion. If all terms are zero it is null.

All RMS values accumulate squared errors over cells, target times and test
realizations, then take the square root. They are not averages of per-frame
ratios. Each equation is reported separately; no misleading average combines
raw stress-rate units with velocity-rate units.

## Interpreting the outcome

- Small CNO and reference residuals: supports bulk discrete consistency at
  this grid/time resolution.
- CNO residual close to a non-small reference residual: prediction inherits
  much of the numerical reference's residual; that is not near-zero PDE error.
- CNO much larger than reference: derivative-level discrepancy remains even
  if field L2 error is small. Differentiation can magnify small field errors.
- CNO residual smaller than reference: not automatically better physics;
  smoothing or nearly zero predictions can reduce the bulk residual too.
  Inspect field accuracy and residual term magnitudes together.

Your earlier reference residual RMS values around 5-8 were dimensionless,
not percentages. The 64-pair training calibration and full 200-realization
test evaluation are not identical samples. This diagnostic can support or
challenge physical consistency but cannot prove that a network has learned
the governing equations or generalizes beyond the training distribution.
Mask, grid spacing, saved time interval and derivative scheme affect the result.

## Files produced

- `residual_summary.json`: configuration, field L2, masks and all eight
  residual equation summaries for both modes.
- `residual_by_channel.csv`: flattened values for comparisons/tables.
- `{mode}_residual_vs_time.png` and `.npz`: CNO/reference/difference RMS for
  each equation, aggregated across the evaluated realizations.
- `{mode}_real0_residual_t*.png`: eight-row reference/CNO/difference maps
  at the selected times for realization zero. Gray cells are excluded.
- Corresponding NPZ maps contain dimensionless residuals, the mask, actual
  time/coordinates, and per-channel scale factors to recover physical units.
  Plots use 99.5th-percentile color limits; raw NPZ values are not clipped.
- `physics_mask.png` and `.npz`.

Use `--no-snapshot-plots` to skip snapshot PNG/NPZ output and ignore plot-time
requests. Global summaries/time curves are still generated. For a different
example use `--plot-realization N`, among the evaluated realizations.

## Verification scope

`python check_residuals.py` verifies a manufactured coupled Biot solution,
source sign, scaling, chunk aggregation and the distinction between a
teacher-forced error and a constant offset across a prediction sequence.
The latter is an explicit counterexample showing that zero bulk residual
difference does not imply zero field error or correct initial/boundary data.
Small synthetic checkpoint/H5 tests verify inference compatibility, both
time windows, snapshot output and metric accounting. CPU-tested; the real
3000/200/200 dataset and production GPU execution are not available here.
