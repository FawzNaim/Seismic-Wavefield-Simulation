# Review and reorganization — 2026-10-28

## Current disposition

Prepared source snapshot; nothing pushed to GitHub and no license selected.
No simulation, training or numerical accuracy result was produced in this pass.

## Requested changes

- Removed `biot_cno_spectral`, `biot_cno_hard_ic` and `cno_absorbing_dataset`.
- Moved two-layer DG solver/generator into `dg/two_layer/`.
- Moved faulted DG core, geometry, materials, interface, generator, check and
  packaged mesh preview into `dg/faulted_three_layer/`.
- Separated faulted CNO training into `faulted_cno_training/`.
- Added independent DG documentation, dependencies and license-status copy.
  The entire `dg/` folder can be copied without training folders.
- Made full generation examples explicit: 3000/200/200 realizations. Existing
  generators still default to 8/2/2 pilots; numerical Python bytes are unchanged.
- Updated active paths, CI, workflow documentation, source inventories and
  two-layer trainer examples. Removed the old mixed solver/training folders.

## Checks and limits

All retained Python sources except the revised packaging checker match the
supplied ZIP byte-for-byte at their original paths. Syntax, full inventory,
static local import closure, sibling hash lookup, documentation link and command
option checks pass. The checker rejects an added syntactically invalid source
in an isolated copy. Numerical dependencies are absent, so runtime execution and
numerical checks remain unverified here. CI results are pending publication.

Historical spectral checkpoint tag strings remain in supported readers to avoid
changing training code. These are compatibility identifiers, not dependencies on
the removed folder. No removed source is bundled in the current tree.

## Attribution retained

The method crosswalk distinguishes published Biot/ADER-DG from project code,
source/mesh choices, static fault geometry and dataset/training extensions.
CNO is described as CNO-inspired because its interpolation/filtering differs from
the original construction. Fifth-order convergence, resolution invariance,
novelty, speedup and exact reproduction are not established for this snapshot.

## Before release

Confirm source provenance and rights, choose license/visibility and repository
owner, execute existing CPU checks and pilot workflows in a recorded environment,
and add quantitative scientific validation before making accuracy claims.
