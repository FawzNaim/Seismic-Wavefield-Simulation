# Workflow compatibility

| Folder | Purpose | Data/checkpoint scope |
|---|---|---|
| dg/two_layer | Independent CUDA DG solver and eight-state HDF5 generator | Two materials; original full-state schema |
| dg/faulted_three_layer | Independent CPU/CUDA DG geometry, solver and HDF5 generator | Three materials; dedicated faulted schema |
| biot_picno_training | Supervised CNO-inspired model; optional Biot losses | Two materials |
| biot_fno_supervised | Supervised FNO comparison | Two materials; fresh FNO weights |
| picno_residual_training | Residual-only objective and logs | Two materials; own checkpoint tag |
| cno_residual_diagnostics | Saved-grid PDE residual checks | Two-material supported checkpoint tags |
| faulted_cno_training | Supervised CNO-inspired model | Three-material schema and dedicated checkpoint tag |
| biot_cno | Earlier pressure/velocity CNO | Three-output workflow, not eight-state interchangeable |

The `dg/` tree can be copied and run without any of the training folders.
Each model subfolder keeps sibling imports and generator hash lookups intact.
No numerical refactor was performed. Run scripts by their documented paths;
this project is not one installable Python package.

Production generation commands explicitly use 3000/200/200 realizations;
generators default to 8/2/2 pilots. Trainers read split sizes from the supplied
HDF5 file. Dataset filenames are conventions, not schema validation.
Do not bypass schema checks or rename checkpoints to force compatibility.

The spectral CNO, hard-IC CNO and sponge-dataset experiment folders were removed.
Some retained readers still accept a historical spectral checkpoint tag for
backward compatibility; this is not an import or dependency on removed code.
The same `--layers` value denotes different operations in CNO and FNO.
