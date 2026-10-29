# Supervised eight-state Biot FNO versus CNO

This package trains an FNO on the EXISTING 3000/200/200 full-state dataset. No new dataset is needed. It matches the original supervised CNO experiment that achieved approximately 1.3% relative error, not the newer residual-only PICNO experiment.

## Mapping and objective

Input: material fields + Gaussian source-location descriptor + target time + four TRUE eight-state history frames. With ten material fields, this is 44 input channels; x/y coordinates are appended internally. Output: eight fields at the next saved time. Neither training nor testing feeds predicted outputs into another input.

As in the CNO, the physical predictor is built from the normalized linear extrapolation 2*U_n - U_(n-1) plus a learned eight-channel correction. Both models have a zero-initialized correction head. FNO starts with its own random weights; it cannot load CNO network weights.

Supervised objective is unchanged from the CNO:

    field_loss + 0.1*spatial_gradient_loss + 0.1*RMS_amplitude_loss

Field loss is 0.7 times SmoothL1 on each channel's strongest 5% target pixels plus 0.3 times global SmoothL1, beta=0.1. Amplitude loss is a field-RMS matching term, not Biot energy conservation. Physics, interface and boundary weights are zero for this comparison. Their optional original implementations are retained, but do not activate them for the supervised comparison.

The data split, train-only normalization, random time sampling (seed+epoch), fixed validation sampling, optimizer (AdamW, lr 2e-4, weight decay 1e-5), gradient clipping, AMP setting, LR scheduler and test metrics follow the original CNO trainer. Width, depth, batch size and all these settings can be copied from the actual CNO checkpoint to avoid assuming which command was used.

## Run

Extract the folder under /path/to/biot-neural-operators/. Replace ONLY the CNO checkpoint placeholder below with the final_picno.pt from the original ZERO-physics supervised run. The filename still says picno because it came from the shared original trainer.

```bash
python -u /path/to/biot-neural-operators/biot_fno_supervised/train_biot_fno.py \
  --data /path/to/biot-neural-operators/biot_picno_coarse_3000_200_200.h5 \
  --match-cno-checkpoint /ACTUAL/PATH/TO/ZERO_PHYSICS_RESULTS/final_picno.pt \
  --output-dir /path/to/biot-neural-operators/fno_supervised_results \
  --device cuda \
  --epochs 150 \
  --no-early-stopping \
  --modes-y 22 \
  --modes-x 22 \
  --plot-times 0.1,0.2,0.3,0.4,0.5 \
  2>&1 | tee /path/to/biot-neural-operators/fno_supervised_training.log
```

--match-cno-checkpoint copies settings and normalization ONLY. It requires the original eight-state CNO tag and zero bulk/interface/boundary physics weights. It validates the dataset configuration. You do not need to recompute normalization. Explicit CLI options override copied settings and such differences are recorded in comparison_setup.json. For strict matching, confirm this file's explicit_overrides is empty, or that each difference is intentional. FNO-specific modes and the output directory are naturally new. Do not use --initialize-from or a residual-only checkpoint for this supervised run.

If you do not want to supply a CNO checkpoint, omit --match-cno-checkpoint and explicitly use the original CNO values. Defaults: history=4, epochs=150, batch=8, width=32, layers=2, padding=8, windows per realization=16, validation windows=0 (all), workers=4, seed=20260927, gradient/amplitude weights=0.1, physics=0. Normalization will then stream over the train split. Use --no-early-stopping for a full 150-epoch budget.

## FNO architecture and parameter counts

Each FNO block computes a learned low-mode Fourier convolution plus a local 1x1 convolution, followed by GELU. A 1x1 lift and two 1x1 projection layers surround the blocks. Padding replicates 8 cells on every edge, as in the CNO. The padding is an architectural treatment, not a change to the dataset's physical boundary conditions.

`layers` means Fourier blocks for FNO, while the CNO uses it for encoder/decoder resolution levels. Their operations/depth are different even when the numeric value is identical. Fourier mode counts have no exact CNO equivalent.

With 44 inputs, 8 outputs, width=32 and layers=2:

| Architecture | Real trainable scalar parameters |
|---|---:|
| Existing CNO | 4,117,256 |
| FNO, modes_y=22 and modes_x=22 | 3,969,864 |
| FNO, modes_y=16 and modes_x=16 | 2,102,088 |

22 is a starting choice that makes model sizes reasonably close (about 3.6% fewer parameters than this CNO); it is not a tuned optimum or exactly matched capacity. If the checkpoint used a different width/layer count, these counts change. Actual counts and architecture are printed/saved. Real and imaginary Fourier weights are stored as separate real entries, so reported counts consistently include both.

Modes_y=22 means 22 bins in each of two nonoverlapping positive/negative y frequency bands; modes_x=22 means x bins 0..21 in the one-sided real FFT. These are spatial frequencies, not temporal Hz. The padded grid must satisfy 2*modes_y <= H and modes_x <= W//2+1; invalid settings raise an error, not silent truncation. FFTs and complex products always use float32/complex64, including under AMP, to support non-power-of-two grids.

Architecture reference: [NeuralOperator's official Fourier layer guide](https://neuraloperator.github.io/dev/theory_guide/fno.html). This is a small explicit dense PyTorch FNO implementation, not a dependency on or exact copy of the current neuralop package.

## Outputs and fair comparisons

- best_fno.pt: checkpoint selected by minimum validation relative_main (p, vx, vy), using the same min-delta rule as the CNO.
- final_fno.pt: last epoch plus optimizer/scheduler/scaler/history for resume.
- training_history.json / training_history.png: same loss and validation-relative-L2 reporting as CNO. A disabled physics panel means not computed, not zero equation residual.
- test_one_step_metrics.json: eight channel relative L2 values, baseline values, relative_main, relative_all and evaluated checkpoint epoch.
- test_one_step_error_vs_time.png / .npz: true-history one-step errors versus target time.
- test_sample0_comparison_t*.png: pressure, vx and vy comparisons at the five requested times, as in the CNO.
- test_sample0_t*.npz: predictions and reference for ALL eight states at each plotted time.
- comparison_setup.json: architecture, parameter count, full settings, matched-CNO configuration and any deliberate CLI overrides.

Relative L2 per channel = sqrt(sum((prediction-reference)^2)/sum(reference^2)) over ALL eligible test times, realizations and grid cells. relative_main is the root mean square of the p/vx/vy relative values; relative_all is the root mean square across eight channel values. Baseline is the same linear extrapolation. These formulas are identical to the CNO's test evaluation.

By default the BEST validation checkpoint is tested after training. To compare final epoch 150 against final epoch 150, pass --evaluate-final consistently in BOTH trainers. Evaluating an already-trained FNO at its final checkpoint:

```bash
python -u /path/to/biot-neural-operators/biot_fno_supervised/train_biot_fno.py \
  --data /path/to/biot-neural-operators/biot_picno_coarse_3000_200_200.h5 \
  --evaluate-only \
  --checkpoint /path/to/biot-neural-operators/fno_supervised_results/final_fno.pt \
  --output-dir /path/to/biot-neural-operators/fno_final_epoch_evaluation \
  --device cuda --batch-size 8 --num-workers 4 \
  --plot-times 0.1,0.2,0.3,0.4,0.5
```

Architecture and original loss/normalization settings are restored on evaluation. To resume a training run in its SAME output folder, replace --match-cno-checkpoint with --resume /.../fno_supervised_results/final_fno.pt. Architecture and invariant training settings are restored unless explicitly supplied, and mismatches are rejected. --epochs is total, not extra epochs. Keep the original seed and sampling unchanged. Batch size and workers may change. CNO weights cannot resume an FNO run.

Concurrent training in separate result folders can use the same HDF5 file read-only. Both jobs on the A6000 share GPU compute, GPU memory and storage bandwidth, so each may slow down. Run separately for meaningful wall-clock comparisons. A single seed/run does not establish an architecture's general superiority.

## Verification

    python check_fno.py

The check verifies explicit real/imaginary spectral multiplication, high-mode truncation, odd/even shapes, mode bounds, zero correction at initialization, and gradient flow to spectral weights. A synthetic schema-compatible dataset additionally tested training/evaluation, settings/statistics reuse from CNO, loss/data/sampling equivalence, checkpoint resume and metrics. These checks are not production accuracy or A6000 timing measurements.

Dependencies: the same PyTorch, NumPy, h5py and Matplotlib environment used for CNO. No extra neural-operator library is needed.
