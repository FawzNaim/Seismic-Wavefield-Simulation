# Biot ADER-DG and neural operators

Two-dimensional inviscid Biot wave simulation, synthetic dataset generation,
and CNO-inspired/FNO learning. Maintainer: Fawz Naim.

**Start with [the independent DG package](dg/README.md).** The entire `dg/`
folder can be copied and used without any neural-operator training folder.
It includes both solvers, dataset generators, dependencies, model documentation,
method attribution and the current license status.

## Repository layout

| Folder | Purpose |
|---|---|
| [dg/](dg/README.md) | Independent DG workflows: two-layer and faulted three-layer |
| [biot_picno_training/](biot_picno_training/README.md) | Eight-state two-layer CNO-inspired model; supervised or optional physics losses |
| [biot_fno_supervised/](biot_fno_supervised/README.md) | Supervised two-layer FNO comparison |
| [faulted_cno_training/](faulted_cno_training/README.md) | Dedicated supervised CNO-inspired trainer for three-material data |
| [picno_residual_training/](picno_residual_training/README.md) | Two-layer residual-only training experiment |
| [cno_residual_diagnostics/](cno_residual_diagnostics/README.md) | Two-layer saved-grid PDE residual diagnostics |
| biot_cno/ | Earlier three-output CNO workflow; not eight-state compatible |
| tools/ | Source inventory and syntax check |


## DG quick start

Install a PyTorch build for your hardware, then, from the repository root:

```bash
python -m pip install -r dg/requirements.txt
mkdir -p data
python dg/faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/geometry_preview.h5 --preview-only
```

Preview creates `data/geometry_preview_figures/geometry_mesh.png`, without
running wave simulations or creating an HDF5 dataset. See [dg/README.md](dg/README.md)
for single-simulation, CPU smoke, full generation and verification commands.
The two-layer simulation entrypoints require CUDA. The faulted entrypoint
supports CPU or CUDA; the documented CPU smoke settings are deliberately small.

## Dataset sizes

The intended full neural-operator dataset has **3000 training / 200 validation /
200 test realizations**, specified explicitly using
`--train 3000 --validation 200 --test 200`. These are independent simulated
realizations, not optimizer iterations or epochs. Generator defaults remain the
8/2/2 pilot. No datasets or trained checkpoints are included.

Two-layer and faulted datasets have distinct schemas. Use the corresponding
trainer; the original FNO and physics-training scripts cannot consume the
three-material dataset unchanged. Training examples now reference 3000/200/200;
trainers read the split sizes in the actual HDF5 file.

## Attribution and scope

Published Biot and ADER-DG methods, including de la Puente et al. (2008), are
credited to their original authors. This project supplies an implementation and
synthetic experiments; it is not the authors' original code or a verified exact
reproduction. See [method provenance](docs/METHOD_AND_PROVENANCE.md),
[method crosswalk](docs/METHOD_CROSSWALK.md) and [references](REFERENCES.md).

The local CNO-inspired model uses antialiased bicubic resizing and oversampled
activations. It is not an exact reproduction of the published sinc-filtered CNO,
and resolution invariance has not been established. Report one-step prediction
errors separately from masked sampled PDE residuals and rollout stability.

## Checks and publication status

```bash
python tools/check_source_snapshot.py
python dg/faulted_three_layer/check_faulted_model.py
python biot_fno_supervised/check_fno.py
```

Only static checks were performed during this reorganization. Numerical checks
above require the appropriate dependencies and have not been rerun here.
See [validation](docs/VALIDATION.md) and [review report](docs/REVIEW_REPORT.md).
No mesh/time convergence or production accuracy claim is made.

A redistribution license remains undecided; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
Nothing has been pushed to GitHub. See [GitHub setup](docs/GITHUB_SETUP.md) and
[CITATION.cff](CITATION.cff). Preserve numerical code and settings while resuming
a dataset; moving intact sibling files does not itself change their content hashes.
