# Residual-only Biot CNO training

This is a new experiment for the existing eight-state dataset and four-true-history CNO mapping. No dataset regeneration is needed. The numerical residual operator is the same type used in the earlier diagnostic: BDF2 in time and fourth-order centered differences in space on the saved uniform grid. It is not the native ADER-DG operator.

## Training and monitoring

At each target time, four TRUE history frames predict one next state. No predicted output is fed into another input. The time derivative uses the current prediction and the two latest TRUE history states (teacher-forced residual). Each training/validation batch reads input history only, not its next-frame ADER-DG target.

For equation c, the dimensionless residual is

    r_c = (T_scale / training_std_c) * [BDF2(U)_c + (A Dx U)_c + (B Dy U)_c - (E U)_c - S_c].

The training objective is

    L = physics_weight * mean_over_8_equations_and_valid_samples(r_c^2).

Optional interface and incoming-boundary penalties can also be activated. Their defaults are zero. There is no supervised field, gradient, amplitude, relative-L2, or ADER-DG reference residual loss. The supplied dataset is still needed for true input histories, material matrices and source. Training statistics are inherited from an initialization checkpoint or computed from the training dataset.

Validation uses frozen end-of-epoch weights and a FIXED set of samples. By default it uses every eligible target time for every validation realization. The training sample changes reproducibly each epoch (16 times per realization by default). Training residuals are online aggregates while weights change, and are labeled accordingly.

Each equation's RMS is sqrt(sum(r_c^2)/N), with N including all evaluated valid cells, times and realizations. The selection metric is the arithmetic average of these EIGHT RMS values. This differs from pooled RMS = sqrt(mean_over_c(RMS_c^2)), which is also saved for clarity. The objective minimizes mean square; selection/stopping use the requested arithmetic mean RMS. No averaging of signed residuals is used.

The script records initialization as epoch 0, then runs up to 200 NEW optimization epochs. With --initialize-from, epoch 1 here follows the weights loaded from the old checkpoint, regardless of its original epoch. Architecture, history length, normalization, and (unless explicitly overridden) residual time scale and mask margins are inherited. Do not change residual scales or mask margins while comparing runs.

## Run on the Linux GPU workstation

Extract this folder into /path/to/biot-neural-operators/. Replace the checkpoint placeholder below with the actual checkpoint from your accurate zero-physics CNO run. That is a weight initialization, not continuation of its old optimizer or stopping metric.

```bash
python -u /path/to/biot-neural-operators/picno_residual_training/train_residual_picno.py \
  --data /path/to/biot-neural-operators/biot_picno_coarse_3000_200_200.h5 \
  --initialize-from /ACTUAL/PATH/TO/ZERO_PHYSICS_RESULTS/final_picno.pt \
  --output-dir /path/to/biot-neural-operators/picno_residual_results \
  --device cuda \
  --epochs 200 \
  --batch-size 8 \
  --num-workers 4 \
  --windows-per-realization 16 \
  --validation-windows 0 \
  --physics-loss-weight 0.01 \
  --learning-rate 2e-5 \
  --residual-dtype float64 \
  --patience 20 \
  --min-delta 1e-5 \
  2>&1 | tee /path/to/biot-neural-operators/picno_residual_training.log
```

The defaults are conservative for precision: the network runs float32 and residual arithmetic float64. The A6000 has relatively slow float64, so timing can differ from earlier supervised training. To favor speed, --residual-dtype float32 and optionally --amp are available; keep a consistent precision during any run/comparison and expect a different numerical floor. GPU execution has not been benchmarked here. batch size 8 and workers 4 match the user's previously successful throughput settings.

You may omit --initialize-from to start random weights (width 32, layers 2, history 4), but this recomputes normalization and discards the accurate starting predictor. With initialization, architecture flags are superseded by the checkpoint's architecture.

With only a bulk physics objective, physics_weight is an overall multiplier, not a tradeoff against a data loss. Increasing it does not guarantee a lower attainable residual, particularly with Adam. The eight equations contribute equally after dimensionless scaling.

## Stopping and saved files

- best_residual.pt: lowest observed mean VALIDATION RMS, including epoch 0 if training never improves it.
- final_residual.pt: most recent weights plus optimizer/scheduler/scaler and history for resuming. It need not be the best model.
- residual_history.csv: opens directly in Excel. One row per epoch, eight validation RMS columns, validation_mean_rms, and online train counterparts. Written after every epoch.
- residual_history.json: all epoch results and sample counts.
- residual_history.png: average validation RMS and eight equation-specific histories, refreshed each epoch.
- residual_summary.json: selected epoch, best validation results and one final evaluation of its residuals on the held-out test set. Test is never used for stopping. --validation-windows also controls the test sampling limit; default 0 means all test times.
- normalization.json: the fixed scales used for the run.

Every new numerical minimum saves best_residual.pt. Patience resets only after cumulative improvement beyond min_delta (absolute RMS units) relative to its previous patience anchor. The learning rate halves after lr-patience=5 plateau epochs. Training stops after patience=20 epochs without sufficient improvement, or at 200 total epochs. This detects an observed plateau, not a guaranteed global minimum. Use --no-early-stopping only if you want all 200 regardless of plateau. An improving average does not guarantee every individual equation improves; inspect all eight curves.

Resume in the SAME results folder using the SAME command/settings, replace --initialize-from with:

    --resume /path/to/biot-neural-operators/picno_residual_results/final_residual.pt

--epochs is the maximum total for this residual-only run, not extra epochs on resume. A run that already exhausted patience will stop again unless --no-early-stopping is supplied. Sampling, scales, mask, precision and stopping parameters must match on resume. Batch size/workers may change.

## Interpretation limits

Only valid interior cells contribute. Source support, interfaces and exterior margins are masked exactly as before. Thus a small bulk loss does not check source excitation, initial conditions, interfaces, or boundaries. This history-conditioned one-step residual is not the residual of an autonomous trajectory. Near-zero is a goal, not a guarantee that this neural architecture can satisfy the particular discrete equations for every supplied history.

The original ADER-DG reference has substantial residual under this saved-grid finite-difference operator. Optimizing this objective can therefore move predictions away from the original numerical wavefields. No reference-error results are generated in this experiment as requested. Low residual alone does not establish a complete physically correct solution. In particular, it does not replace independent IC/source/BC verification.

The new checkpoint tag intentionally differs from old supervised trainers. Do not use their --resume. This trainer's own per-epoch outputs provide the requested monitoring; old diagnostic tools may reject its new tag until extended.

## Local verification

    python check_residual_training.py

Checks cover a manufactured coupled PDE, BDF2 scaling, source sign, differentiable physics loss and RMS aggregation. A separate synthetic HDF5 smoke run verified training, initialization, history export, early stopping, best-checkpoint selection, target-independent batches and resume. These are software checks, not results on the user's production dataset. Requires the same PyTorch, NumPy, h5py and Matplotlib environment as the original trainer.
