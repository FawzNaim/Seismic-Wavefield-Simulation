# Faulted three-layer Biot DG workflow

This standalone DG workflow implements a fixed synthetic cross-section with
three effective materials. It runs without neural-operator training code.
Commands on this page are run from the parent `dg/` directory. See
[installation and common instructions](../README.md).

## Exact approved setup

Domain: x=0..1500 m; depth=0..1400 m. Public acquisition coordinates are **(x, depth), depth increasing downward**.

| Item | Position / configuration |
|---|---|
| Source | (250, 1200) m, 50 Hz Ricker, delay 0.04 s |
| Receiver R1 | (1000, 1000) m |
| Receiver R2 | (1000, 600) m |
| Receiver R3 | (1000, 200) m |
| Material 0 | Shale |
| Material 1 | Shaly-sandstone, 30% shale / 70% sandstone |
| Material 2 | Sandstone |
| Before-faulting contact depths | 350 m, 700 m |
| F1, F3 | 75 degrees right-dipping |
| F2, F4 | 75 degrees left-dipping |
| F5 | 55 degrees right-dipping |
| Fault upper/lower tip depths | 200 m / 950 m |
| Maximum vertical throw per fault | 60 m, tapering to zero at the tips |

Exact fault top x positions are F1=25, F2=457, F3=487, F4=919, F5=949 m. These are the positions used in the approved diagram. Dip is measured from horizontal. Hanging walls move downward. Throw at depth d between the tips is 60*sin(pi*(d-200)/750)^2 m; it is zero outside. Geometry is FIXED across realizations, while end-member material properties vary reproducibly. Variation in fault positions/dips is not introduced into this dataset.

The numerical kernel uses y upward, so y=1400-depth. Therefore the source is internally (250,200); receivers internally (1000,400), (1000,800), (1000,1200). Plots always use depth downward. The HDF5 file records both coordinate conventions explicitly; fields are stored in increasing solver y order to keep the existing neural-network array convention. `coords/depth_m` is correspondingly decreasing with array row number.

Initial conditions are zero in all eight states. Source strength remains the earlier **assumed** point-source coefficient 2e10 (not an imposed pressure), applied with opposite signs to syy and p. It is not a newly sourced published amplitude.

## Material mixing assumption

The 30/70 middle layer is ONE isotropic effective material. For shale fraction f=0.3:

    Hill(a,b) = 0.5 * [f*a + (1-f)*b + 1/(f/a + (1-f)/b)].

Hill averaging is applied to drained bulk modulus Kd, shear modulus mu and grain bulk modulus Ks; dry lambda is reconstructed as Kd-2*mu/3. Grain density, porosity and tortuosity use arithmetic weighted averages. Permeability uses a geometric mean. Both rocks share the same sampled fluid density and bulk modulus. Biot alpha, M, inertial mass and wave speeds are recomputed from the resulting properties, and nonphysical draws are rejected.

This is an explicit MODELING APPROXIMATION for this synthetic shaly-sandstone, not a uniquely determined rock-physics law or a claim that a 30/70 real rock must have these properties. Permeability does not affect the inviscid wave equations here because viscosity remains zero. Mixture fraction is fixed at 0.3; the middle layer is correlated with the two sampled end members.

End-member perturbations match the earlier data strategy: solid density ±5%; dry lambda/mu and grain modulus ±10%; porosity ±0.02 absolute; tortuosity ±10%; common fluid density ±2%, fluid bulk modulus ±10%; fixed end-member permeabilities. Split/sample seeds are independent and deterministic.

## Numerical formulation

- Triangle's constrained quality mesher builds triangles whose EDGES follow both displaced contacts and all five finite fault traces. Element material assignment is checked at interior barycentric samples; unmatched interior mesh edges are rejected.
- Default maximum triangle area is 120 m², minimum-angle request 28 degrees. The current default mesh has **27,253 triangles**. This is NOT the old 17,280-triangle structured coarse mesh. Mesh library versions can change triangulations; the mesh hash is validated when resuming.
- Polynomial degree 4 and the existing local Taylor ADER predictor give the intended O5 scheme on homogeneous smooth elements. Interfaces and point-source singularities do not imply global fifth-order convergence. Time step uses the mesh CFL estimate and the fastest material wave speed, with safety factor 0.9.
- Same-material faces retain the original Rusanov-type flux with a maximum adjacent wave-speed bound.
- At material-changing faces, a characteristic interface solve enforces continuity of total traction (two components), solid velocity (two components), pressure and normal relative-fluid flux. Incoming characteristics are solved on both sides after scaling for conditioning. This generalizes transmission to arbitrary contact normals.
- Faults represent static offsets with perfectly bonded, open-pore contacts. There is no active fault slip, friction law, opening, damage zone, fault gouge or distinct fault permeability. A fault segment within one homogeneous material has no extra constitutive scattering effect merely because a line is drawn there.
- The exterior remains a zero-exterior-trace numerical boundary, as in the earlier basic generator. Absorption is approximate; there is no PML or newly added sponge.
- The Ricker temporal source integral is computed from its exact antiderivative.

The source mesh element and receiver points are located on the triangular mesh. Grid values are sampled from DG polynomial fields, not produced by a separate finite-difference solver.

## Files

| File | Role |
|---|---|
| `aderdg_core.py` | Biot material matrices, modal basis, DG predictor/update and point sampling |
| `fault_geometry.py` | Contacts, finite static faults, coordinate conversion and constrained mesh |
| `materials.py` | Three effective materials and admissible random sampling |
| `interface_flux.py` | Bonded/open-pore characteristic transmission on material-changing faces |
| `generate_faulted_biot_dataset.py` | Preview, simulation, HDF5 writing, resume and verification |
| `check_faulted_model.py` | Existing CPU geometry/material/interface/DG checks |

The generator configures the core and installs interface fluxes. Keep these six
Python files together. Source hashes are calculated from their expected sibling
filenames; the numerical routines are unchanged by this folder move.

## Run

From `dg/`, first preview:

```bash
python faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_preview.h5 --preview-only
```

A pilot uses explicit 8/2/2 counts. The full dataset uses 3000/200/200:

```bash
python -u faulted_three_layer/generate_faulted_biot_dataset.py \
  --output data/faulted_biot_3000_200_200.h5 \
  --train 3000 --validation 200 --test 200 --device cuda
```

See [DG instructions](../README.md) for a reduced CPU smoke command and verification.
Do not reuse an existing pilot filename for production settings. Rerun the exact
same command and unchanged numerical code to resume; completed realizations are
skipped and an incomplete realization restarts. Source hashes and mesh hash are
checked, but HDF5 completion flags do not guarantee disk-failure recovery.

## HDF5 data

States: `[sxx, syy, sxy, vx, vy, p, qx, qy]`. Stress and pressure are Pa; solid velocities and relative-fluid flux q are m/s. q is porosity times fluid-minus-solid velocity, not a volume in m³.

Each split stores `inputs[N,10,120,128]`, `wavefields[N,251,8,120,128]`, `receiver_wavefields[N,251,3,8]`, `matrix_A/B/E[N,3,8,8]`, sampled material parameters, completion flags and run timings. Ten material-input channels keep the old names; material_id now has THREE values. Shared mesh connectivity, sampled material IDs, source descriptor, projected DG source and both acquisition-coordinate conventions are stored explicitly.

## Checks and limits

```bash
python faulted_three_layer/check_faulted_model.py
```

This existing CPU check tests acquisition conversion, geometry/material
admissibility, characteristic interface continuity, Ricker integration, zero
state and constant-state preservation away from the exterior. It was not rerun
during reorganization. Earlier smoke-test claims in development documentation
are historical, not new production validation. No convergence, transmission-error
or boundary-reflection results are supplied for a scientific accuracy claim.

Training belongs to the separate `faulted_cno_training/` folder in the complete
repository. It is not required to run this DG workflow. Full data are eight-state,
three-material; older two-layer FNO/physics trainers need adaptation.

Method references: [Triangle constrained meshing](https://www.cs.cmu.edu/~quake/triangle.quality.html)
and [Hill-average background](https://www.ctcms.nist.gov/potentials/atomman/tutorial/3.1._ElasticConstants_class.html).
The particular mixture recipe is a project modeling assumption.
