# References and attribution

- de la Puente, J., Dumbser, M., Käser, M., and Igel, H. (2008).
  *Discontinuous Galerkin methods for wave propagation in poroelastic media.*
  Geophysics, 73(5), T77–T97. https://doi.org/10.1190/1.2965027
- de la Puente, J. (2008). *Seismic Wave Simulation for Complex Rheologies on
  Unstructured Meshes.* Dissertation, Ludwig-Maximilians-Universität München.
  https://edoc.ub.uni-muenchen.de/8074/
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K.,
  Stuart, A., and Anandkumar, A. (2021). *Fourier Neural Operator for Parametric
  Partial Differential Equations.* ICLR. https://arxiv.org/abs/2010.08895
- Raonić, B., Molinaro, R., De Ryck, T., Rohner, T., Bartolucci, F., Alaifari, R.,
  Mishra, S., and de Bézenac, E. (2023). *Convolutional Neural Operators for robust
  and accurate learning of PDEs.* NeurIPS 36. https://arxiv.org/abs/2302.01178
  Architectural background; local filtering differs (see crosswalk).
- Fourier operator implementation background:
  https://neuraloperator.github.io/dev/theory_guide/fno.html

The DG method is credited to the published work; this repository's implementation
and synthetic extensions are described in `docs/METHOD_AND_PROVENANCE.md`.
An equation-by-equation paper/thesis crosswalk has not been completed. Do not
infer that every parameter, numerical flux, or modeling choice was taken from
these references. No paper PDFs or original-author source release are bundled.

PyTorch, NumPy, SciPy, h5py, Matplotlib, and Triangle are external dependencies
with their own licenses. The FNO/CNO implementations are local implementations,
not a claim to distribute the original authors' code. The neural-operator papers
above credit the architecture families; they do not certify equivalence of these
local models or transfer their theoretical guarantees. See the scoped
[method crosswalk](docs/METHOD_CROSSWALK.md).
