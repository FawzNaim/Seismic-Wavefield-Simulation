# Faulted three-material supervised CNO training

`train_faulted_cno.py` adapts the original full-state CNO input checks to three materials. It retains four true histories, linear extrapolation plus learned correction, the original supervised field/gradient/amplitude objective, AdamW, validation checkpoint selection and all-test one-step L2 metrics. Default physics weights are zero; positive physics/interface/boundary weights are rejected in this entrypoint. The old horizontal-interface physics penalty would be invalid for the faulted layout, so it is not silently reused.

Do not train until all splits are complete. Fresh weights are needed; the old two-layer models were trained on a different material/geometry distribution.

```bash
python -u /path/to/biot-neural-operators/faulted_cno_training/train_faulted_cno.py \
  --data /path/to/biot-neural-operators/faulted_biot_3000_200_200.h5 \
  --output-dir /path/to/biot-neural-operators/faulted_cno_results \
  --device cuda --history-frames 4 \
  --epochs 150 --no-early-stopping \
  --batch-size 8 --num-workers 4 --width 32 --layers 2 \
  --windows-per-realization 16 --validation-windows 0 \
  --physics-loss-weight 0 \
  --plot-times 0.1,0.2,0.25,0.3,0.4,0.5 \
  2>&1 | tee /path/to/biot-neural-operators/faulted_cno_training.log
```

Training outputs best_cno.pt, final_cno.pt, training history, eight-channel test L2 metrics and plots. --resume uses final_cno.pt in the same results folder with unchanged architecture/loss/sampling settings; --epochs is total, not extra epochs. This trainer follows the original checkpoint selection by validation relative_main, rather than the residual-only stopping experiment. The old FNO/residual-only scripts are not yet three-material/fault-interface-compatible.


## Placement and data generation

Keep `train_faulted_cno.py`, `cno_model.py` and `biot_physics.py` together in this
folder. The model is CNO-inspired, using antialiased bicubic interpolation.
It has no imports from DG solver folders; it reads the generated HDF5 file.

Generate data separately using [the independent DG package](../dg/README.md),
then pass the actual path with `--data`. Generation instructions are relative
to `dg/`; training commands above show absolute repository paths. If generated
under `dg/data/`, that is the dataset path to supply. Install PyTorch and the
root requirements for training. Neither old two-layer weights nor two-layer
physics penalties are compatible with this faulted schema unchanged.

No training was rerun during reorganization. Existing performance claims on
two-layer data do not establish accuracy for this model.
