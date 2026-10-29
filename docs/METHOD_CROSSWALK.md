# Published methods and this implementation

Reviewed 2026-10-28. This is a scoped literature-to-code comparison, not a complete
equation-by-equation verification. Links identify primary sources; code locations
identify the implementation inspected. No numerical equivalence test was run.

## DG attribution

Primary sources: [de la Puente et al. (2008)](https://doi.org/10.1190/1.2965027)
and [de la Puente thesis](https://edoc.ub.uni-muenchen.de/8074/1/DelaPuente_Josep.pdf),
Chapters 1 and 4. Detailed benchmark values below were checked against the thesis,
not independently against the full journal article.

| Topic | Published basis / thesis location | Local code and scope |
|---|---|---|
| Governing equations | Biot velocity–stress system; Chapter 4 | `BiotMaterial.matrices()` in both solver cores: 2D, isotropic, eight states; not the general 3D anisotropic system. |
| Time integration | ADER; stiff space-time treatment, Chapter 4 | `GPUADERDG.predictor_integral()` recursively applies the local spatial operator. The predictor has no viscous E term; no stiff viscous solver is supplied. |
| Numerical flux | Godunov and Rusanov, Sections 1.4.1–1.4.2 | Same-material Rusanov; `dg/faulted_three_layer/interface_flux.py` adds a characteristic bonded/open-pore transmission solve. Equivalence to a published heterogeneous flux is unverified. |
| Two-layer benchmark | Section 4.7.4, Table 4.6: inviscid shale/sandstone; 1500 × 1400 m, interface y=700 m | `dg/two_layer/two_layer_coarse_gpu_colab.py`: matching nominal material/acquisition parameters, with a project mesh and source coefficient. |
| Acquisition | Section 4.7.4: source (750,900) m; 50 Hz, delay 0.04 s; receivers (950,750), (950,650), (950,500) m | These constants occur in the two-layer core. Its assumed strength 2e10 is not verified from the source. Faulted acquisition is a separate project choice. |
| Resolution | Section 4.7.4: 132,014 triangles; nominal O5; final time 0.5 s | Two-layer default: 17,280 triangles, degree 4. Matching nominal order does not establish matching accuracy. Faulted geometry uses a different constrained mesh. |
| Boundaries | No boundary-equivalence claim checked here | Zero exterior trace in `GPUADERDG.step()` is approximate absorption; no sponge or PML is included. |

Do not call the implementation a replication of the publication's full method or
its validation. Neither the Taylor predictor nor exact integration of the scalar
Ricker wavelet proves fifth-order behavior for the forced, heterogeneous problem.
The source is added after the predictor/flux update; source treatment merits a
separate temporal-convergence benchmark.

## Neural operators and project extensions

[FNO: Li et al.](https://arxiv.org/abs/2010.08895) supplies Fourier-operator
background. `biot_fno_supervised/fno_model.py` implements truncated 2D Fourier
convolutions and local branches. Coordinates, padding, history conditioning,
losses and correction-head initialization are local choices.

[CNO: Raonić et al.](https://arxiv.org/abs/2302.01178), Section 2 and Appendix C.1.4,
supplies convolutional-operator background. `biot_picno_training/cno_model.py` and
its workflow copies use antialiased bicubic interpolation, oversampled activation
and an encoder/decoder. Describe these as CNO-inspired; equivalence to the
published filters and resolution guarantees is unverified.

GPU execution, reproducible material sampling, resumable HDF5 generation,
eight-state history-to-next-frame training, loss variants, residual diagnostics,
and the static faulted three-material experiment are project implementation and
experimental extensions. This list does not establish scientific novelty.
Mesh construction and preprocessing use CPU NumPy/Triangle; GPU-oriented does
not mean all stages run on a GPU.

The diagnostics use saved-grid BDF2/fourth-order spatial differences, not the DG
operator. Bulk residual masks omit source/interface/exterior regions. Report
masked coverage, reference residual and held-out prediction error separately.
Teacher-forced one-step metrics do not demonstrate autonomous rollout stability.
