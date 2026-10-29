# Method, provenance, and scope of contribution

## Attribution

The scientific basis is Biot poroelastic wave theory and published ADER-DG
methods, including de la Puente, Dumbser, Käser and Igel (2008). Their numerical
method is not claimed as a new invention of this project. This archive contains
the implementation developed for this research workflow, with AI coding assistance;
it does not contain a source release supplied by those paper authors.

The supplied project documentation reports development without access to the
authors' original implementation; this review cannot independently establish
the complete development history.
This does not establish that no relevant implementation exists anywhere, nor
establish novelty by itself. Assess novelty through a literature/software review
and quantitative evidence before making a first-of-its-kind claim.

The exact origin of any imported local solver files should be checked by the
maintainer before selecting a license. The archive records the local source-file
hashes in `SOURCE_MANIFEST.json`; this is an inventory, not independent proof of
copyright ownership or complete historical provenance.

## Governing system and conventions

The state vector is U = [sxx, syy, sxy, vx, vy, p, qx, qy].

    U_t + A U_x + B U_y = E U + S

Stress and pressure are in Pa. Solid velocity and q are in m/s, with
q = phi * (v_fluid - v_solid). The solver uses y upward. Public coordinates for
the faulted example use depth downward, with y = 1400 - depth.

Current datasets use zero viscosity, so E is zero for viscous drag. Permeability
is recorded but cannot be learned as an active viscous attenuation parameter
from these inviscid simulations. These are not consolidation-flow datasets.

## Numerical implementation

- Modal DG on triangular elements; default degree 4 (15 basis functions/state/element).
- Local Taylor ADER time predictor, with intended fifth-order behavior on smooth
  homogeneous problems. No global fifth-order convergence claim is made for the
  full source/interface geometry.
- Element-local derivatives and face coupling; no symmetric interior-penalty
  term is used in the first-order stress/velocity formulation.
- Rusanov-type same-material fluxes.
- The new faulted workflow uses a characteristic transmission solve on
  material-changing faces, enforcing continuity of traction, solid velocity,
  pressure and normal relative-fluid flux. This is a change from the earlier
  two-layer flux treatment, not an assertion of exact equivalence to the paper.
- Zero-exterior-trace outer treatment in the base generators is approximate
  absorption, not a PML. No sponge or PML implementation is included in the current package.
- DG fields are sampled onto a uniform grid for neural-operator training;
  the 120 x 128 grid is not the DG element mesh.

## Project-specific implementation and experiments

Documentable contributions in this archive include a GPU-oriented PyTorch
implementation, deterministic resumable dataset generators, full eight-state
training/evaluation workflows, one-step CNO/FNO comparisons, discrete residual
diagnostics, residual-only training, and a fault-conforming three-material model.
These are implementation contributions; scientific novelty remains to be evaluated.

## Assumptions introduced for this project

The source strength 2e10 is an assumed coefficient, not an amplitude verified from
the paper. The new example uses a 50 Hz Ricker source at (250, 1200 m depth),
three receivers, three effective materials, five finite normal faults, and zero
initial fields. The geometry and acquisition are fixed across realizations.
The middle material has 30% shale and 70% sandstone with documented effective
property mixing. Faults are static bonded/open-pore offsets, without friction,
active slip, gouge, a damage zone, or separate permeability.

Material perturbations, fault positions/dips/throws, meshing choices, sampling,
source coefficient, and model/loss hyperparameters are project choices. They
must not be attributed wholesale to the publication. See the detailed faulted
workflow README for exact values.

## Physics residual versus supervised error

The diagnostics evaluate a sampled strong-form residual using BDF2 in time and
fourth-order centered spatial differences, with stencil/source/interface/boundary
masks. This is a different discrete operator from the internal DG update.
The dimensionless residual uses a time scale and channel normalizations.
A nonzero residual for sampled DG reference fields therefore is not automatically
a failure of the DG solver. Neither a small sampled residual nor a small one-step
prediction error alone establishes physical correctness over full trajectories.

## Architecture attribution and reviewed evidence

The local CNO is CNO-inspired: `resize_filtered` uses PyTorch antialiased bicubic
interpolation and `FilteredActivation` oversamples by a factor of two around a
LeakyReLU. These choices differ from the published CNO filtering construction.
Do not inherit continuous-discrete equivalence, resolution-invariance or
super-resolution claims without testing this implementation. The FNO uses
truncated Fourier modes plus a local linear branch, with project-specific
padding, coordinates and a zero-initialized correction head.

See [METHOD_CROSSWALK.md](METHOD_CROSSWALK.md) for the scoped published-method
comparison and source locations. Historical solver comments such as “matches the
published numerical configuration” and “unreleased mesh” are retained to preserve
Python bytes. They are not evidence of exact reproduction or of source-code
unavailability. The crosswalk states the reviewed scope; the source strength,
mesh, flux details and validation results require separate treatment.
