# Validation status

## Current reorganization — 2026-10-28

The spectral CNO, hard-IC CNO and sponge-dataset workflows were removed. DG files
were moved to `dg/two_layer/` and `dg/faulted_three_layer/`. The faulted trainer
and its two helper modules were moved to `faulted_cno_training/`.

All retained solver, generator, numerical-check, training and model Python files
are byte-identical to the corresponding paths in the supplied ZIP. Only the
standard-library source checker was revised. All included Python files are parsed
with AST. The rebuilt inventory records current paths/hashes and retains available
original source paths/hashes for provenance. It is an inventory, not proof of rights.

Static import and sibling-file checks establish that the DG entrypoints depend
only on their own DG folder and third-party libraries, not neural-operator code.
Documentation paths, CLI option names and YAML are checked. The source checker
is also exercised on a copied tree containing an added syntax error.

No numerical checks, wave simulations, dataset generation or training were run
in this pass. PyTorch, h5py and Triangle are unavailable in the review runtime.
The small CPU example is a documented command, not a claimed successful run.
GitHub Actions is configured but has not been run for this snapshot.

## Earlier checks and reported results

The original packaging pass parsed 50 Python files and checked a 60-entry source
inventory. This predates the requested removals and is not the current file count.
The supplied archive reports earlier geometry, interface, finite CPU simulation,
resume and trainer smoke checks; their logs and data are not bundled here, and
these results were not independently reproduced during review/reorganization.
Reported approximately 1.3% CNO and 3% FNO errors apply to earlier two-layer
supervised experiments, not a reproduced result or a faulted-model baseline.

## Scientific verification still needed

Record homogeneous analytical comparisons, spatial/time convergence including
source treatment, reflection/transmission benchmarks, boundary reflections and
full-duration production-mesh checks. Test CPU/GPU and precision sensitivity.
Report saved-grid residuals separately from DG accuracy and prediction errors.
Teacher-forced one-step errors alone do not establish stable recursive rollout.
