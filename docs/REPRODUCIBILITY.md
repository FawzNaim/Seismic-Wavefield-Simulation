# Reproducibility checklist and data handling

For each experiment record:

1. Git commit ID, uncommitted changes, exact command, and dataset schema/configuration.
2. Python/dependency versions (`python -m pip freeze > environment-freeze.txt`),
   PyTorch/CUDA versions, GPU/CPU model, and mesher version.
3. Mesh hash, element count, polynomial degree, time-step controls, frame interval,
   final time, coordinate convention, source, boundaries, and interface treatment.
4. Split sizes/seeds, material sampling and normalization statistics.
5. Network width/layers/modes, history mapping, complete loss weights, optimizer,
   scheduler, precision, early-stopping settings, and checkpoint epoch/selection rule.
6. Held-out channel relative-L2 errors separately from dimensionless PDE residuals.

Keep HDF5 data and trained weights outside source control. Archive them separately
with checksums and a persistent data identifier when available. Code alone does
not make a large experiment reproducible without its settings and environment.
No datasets, trained checkpoints, uploaded papers, or user result spreadsheets are
included in this source archive.

Dataset resumption skips only completion-marked realizations. Keep code, geometry,
settings, counts and output path consistent. HDF5 completion markers do not
provide unconditional protection against interrupted disk writes. Verification
can be I/O intensive. Do not clear HDF5 status flags or rewrite metadata blindly.
