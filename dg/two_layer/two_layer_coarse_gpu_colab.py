"""GPU-vectorized coarse O5 ADER-DG shale/sandstone model.

This is a self-contained PyTorch/CUDA implementation of the heterogeneous
2-D poroelastic benchmark in de la Puente et al. (2008), Figures 9-10, and
de la Puente's thesis, Section 4.7.4 and Figures 4.9-4.10.

Colab usage
-----------
1. Select Runtime -> Change runtime type -> GPU.
2. Upload this file.
3. Run:

       %run /content/two_layer_paper_gpu_colab.py

The paper did not publish its mesh coordinates/connectivity or absolute source
strength. This script creates a deterministic interface-conforming irregular
mesh with 17,280 interface-conforming triangles. It matches the published numerical
configuration, but cannot recreate the authors' unreleased mesh bit-for-bit.

The solver uses CUDA float32 by default. Validate it against the CPU solver on
a reduced mesh before treating large-run output as a reference dataset.
"""

from dataclasses import dataclass
from math import factorial
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.tri as mtri
import numpy as np
import torch


# =============================================================================
# 1. USER CONTROLS AND PUBLISHED CONFIGURATION
# =============================================================================

DOMAIN_X_M = (0.0, 1500.0)
DOMAIN_Y_M = (0.0, 1400.0)
INTERFACE_Y_M = 700.0

# Coarse interface-conforming mesh for large neural-operator datasets.
# 96*90*2 = 17,280 triangular elements.
NX = 96
NY = 90
POLYNOMIAL_DEGREE = 4             # O5 ADER-DG
FINAL_TIME_S = 0.5
PAPER_DT_S = 2.34e-5

SOURCE_X_M = 750.0
SOURCE_Y_M = 900.0
SOURCE_FREQUENCY_HZ = 50.0
SOURCE_DELAY_S = 0.04

# The publication specifies equal/opposite sigma_yy and p components but not
# their absolute magnitude. Normalize receiver traces for paper comparison.
POINT_SOURCE_STRENGTH = 2.0e10

RECEIVERS_M = np.array(
    [[950.0, 750.0], [950.0, 650.0], [950.0, 500.0]], dtype=float
)
SNAPSHOT_TIMES_S = [0.25, 0.50]

OUTPUT_NX = 128
OUTPUT_NY = 120
FIGURE_DPI = 180

# For persistent Colab checkpoints, replace this with a mounted Drive path,
# for example Path('/content/drive/MyDrive/biot_gpu_results').
OUTPUT_DIR = Path("coarse_gpu_results")
OUTPUT_FILE = "coarse_gpu_two_layer_output.npz"
CHECKPOINT_FILE = "coarse_gpu_checkpoint.pt"
CHECKPOINT_EVERY_STEPS = 1000
RESUME_IF_CHECKPOINT_EXISTS = True

MESH_RANDOM_SEED = 20260924
MESH_JITTER_FRACTION = 0.15

# float32 is recommended on Colab T4/L4 GPUs. float64 is much slower on T4.
TORCH_DTYPE = torch.float32
PROGRESS_EVERY_STEPS = 100
SAMPLING_BATCH_SIZE = 65536


# =============================================================================
# 2. QUADRATURE AND ORTHONORMAL MODAL BASIS
# =============================================================================

def triangle_quadrature(n):
    x, w = np.polynomial.legendre.leggauss(n)
    a, wa = 0.5 * (x + 1.0), 0.5 * w
    points, weights = [], []
    for i, ai in enumerate(a):
        for j, bj in enumerate(a):
            points.append((ai, (1.0 - ai) * bj))
            weights.append(wa[i] * wa[j] * (1.0 - ai))
    return np.asarray(points), np.asarray(weights)


def line_quadrature(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class ModalBasis:
    """Orthonormal polynomials of total degree <= degree on a triangle."""

    def __init__(self, degree):
        self.degree = int(degree)
        self.exponents = [
            (i, total - i)
            for total in range(self.degree + 1)
            for i in range(total + 1)
        ]
        self.size = len(self.exponents)

        gram = np.empty((self.size, self.size))
        for i, (a, b) in enumerate(self.exponents):
            for j, (c, d) in enumerate(self.exponents):
                gram[i, j] = (
                    factorial(a + c)
                    * factorial(b + d)
                    / factorial(a + b + c + d + 2)
                )
        self.transform = np.linalg.inv(np.linalg.cholesky(gram))

        points, weights = triangle_quadrature(self.degree + 3)
        phi, dxi, deta = self.evaluate(points, derivatives=True)
        self.d_xi = phi.T @ (weights[:, None] * dxi)
        self.d_eta = phi.T @ (weights[:, None] * deta)

    def evaluate(self, points, derivatives=False):
        points = np.atleast_2d(points)
        xi, eta = points[:, 0], points[:, 1]
        monomials = np.empty((len(points), self.size))
        dxi = np.empty_like(monomials)
        deta = np.empty_like(monomials)
        for k, (a, b) in enumerate(self.exponents):
            monomials[:, k] = xi**a * eta**b
            dxi[:, k] = 0.0 if a == 0 else a * xi ** (a - 1) * eta**b
            deta[:, k] = 0.0 if b == 0 else b * xi**a * eta ** (b - 1)
        phi = monomials @ self.transform.T
        if not derivatives:
            return phi
        return phi, dxi @ self.transform.T, deta @ self.transform.T


# =============================================================================
# 3. ISOTROPIC BIOT MATERIALS: PAPER TABLE 5 / THESIS TABLE 4.6
# =============================================================================

@dataclass(frozen=True)
class BiotMaterial:
    name: str
    rho_s: float
    rho_f: float
    lambda_matrix: float
    mu_matrix: float
    bulk_solid: float
    bulk_fluid: float
    porosity: float
    permeability: float
    tortuosity: float
    viscosity: float = 0.0

    @property
    def bulk_matrix(self):
        return self.lambda_matrix + 2.0 * self.mu_matrix / 3.0

    @property
    def alpha(self):
        return 1.0 - self.bulk_matrix / self.bulk_solid

    @property
    def coupling_M(self):
        denominator = (
            1.0
            - self.porosity
            - self.bulk_matrix / self.bulk_solid
            + self.porosity * self.bulk_solid / self.bulk_fluid
        )
        if denominator <= 0.0:
            raise ValueError(f"Nonphysical Biot denominator for {self.name}")
        return self.bulk_solid / denominator

    @property
    def bulk_density(self):
        return (1.0 - self.porosity) * self.rho_s + self.porosity * self.rho_f

    @property
    def inertial_m(self):
        return self.rho_f * self.tortuosity / self.porosity

    @property
    def mass_determinant(self):
        return self.bulk_density * self.inertial_m - self.rho_f**2

    def matrices(self):
        """A and B for Q_t + A Q_x + B Q_y = 0 (inviscid case)."""
        A = np.zeros((8, 8))
        B = np.zeros((8, 8))
        M = self.coupling_M
        alpha = self.alpha
        mu = self.mu_matrix
        lambda_u = self.lambda_matrix + M * alpha**2
        H, C = lambda_u + 2.0 * mu, lambda_u
        rho, rho_f, m = self.bulk_density, self.rho_f, self.inertial_m
        delta = self.mass_determinant

        A[0, 3], A[0, 6] = -H, -alpha * M
        A[1, 3], A[1, 6] = -C, -alpha * M
        A[2, 4] = -mu
        A[3, 0], A[3, 5] = -m / delta, -rho_f / delta
        A[4, 2] = -m / delta
        A[5, 3], A[5, 6] = alpha * M, M
        A[6, 0], A[6, 5] = rho_f / delta, rho / delta
        A[7, 2] = rho_f / delta

        B[0, 4], B[0, 7] = -C, -alpha * M
        B[1, 4], B[1, 7] = -H, -alpha * M
        B[2, 3] = -mu
        B[3, 2] = -m / delta
        B[4, 1], B[4, 5] = -m / delta, -rho_f / delta
        B[5, 4], B[5, 7] = alpha * M, M
        B[6, 2] = rho_f / delta
        B[7, 1], B[7, 5] = rho_f / delta, rho / delta
        return A, B

    def wave_speeds(self):
        A, _ = self.matrices()
        values = np.sort(np.real_if_close(np.linalg.eigvals(A)).real)
        return values[values > 1.0e-7]


SHALE = BiotMaterial(
    "shale", 2210.0, 1040.0, 3.96e9, 3.96e9, 7.6e9, 2.5e9,
    0.16, 1.0e-13, 2.0, 0.0,
)
SANDSTONE = BiotMaterial(
    "sandstone", 2500.0, 1040.0, 12.0e9, 12.0e9, 40.0e9, 2.5e9,
    0.20, 600.0e-15, 2.0, 0.0,
)
MATERIALS = (SHALE, SANDSTONE)


# =============================================================================
# 4. DETERMINISTIC IRREGULAR, INTERFACE-CONFORMING MESH
# =============================================================================

@dataclass
class TriangularMesh:
    vertices: np.ndarray
    elements: np.ndarray
    neighbors: np.ndarray
    neighbor_faces: np.ndarray

    @property
    def element_vertices(self):
        return self.vertices[self.elements]


def build_paper_mesh():
    x = np.linspace(DOMAIN_X_M[0], DOMAIN_X_M[1], NX + 1)
    y = np.linspace(DOMAIN_Y_M[0], DOMAIN_Y_M[1], NY + 1)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack((X.ravel(), Y.ravel()))

    hx = (DOMAIN_X_M[1] - DOMAIN_X_M[0]) / NX
    hy = (DOMAIN_Y_M[1] - DOMAIN_Y_M[0]) / NY
    amplitude = MESH_JITTER_FRACTION * min(hx, hy)
    rng = np.random.default_rng(MESH_RANDOM_SEED)

    for j in range(1, NY):
        on_interface = np.isclose(y[j], INTERFACE_Y_M)
        for i in range(1, NX):
            vertex = j * (NX + 1) + i
            vertices[vertex, 0] += rng.uniform(-amplitude, amplitude)
            if not on_interface:
                vertices[vertex, 1] += rng.uniform(-amplitude, amplitude)

    def vertex_id(i, j):
        return j * (NX + 1) + i

    vertices = vertices.tolist()
    elements = []
    # The special paper-resolution center splits were used only to reach
    # exactly 132,014 triangles. They are unnecessary for the coarse mesh.
    center_split_cells = set()
    for j in range(NY):
        for i in range(NX):
            v00, v10 = vertex_id(i, j), vertex_id(i + 1, j)
            v01, v11 = vertex_id(i, j + 1), vertex_id(i + 1, j + 1)
            if (i, j) in center_split_cells:
                center = len(vertices)
                corners = np.asarray(
                    [vertices[v00], vertices[v10], vertices[v11], vertices[v01]]
                )
                vertices.append(np.mean(corners, axis=0).tolist())
                elements.extend(
                    [
                        (v00, v10, center),
                        (v10, v11, center),
                        (v11, v01, center),
                        (v01, v00, center),
                    ]
                )
            elif (i + j) % 2 == 0:
                elements.extend([(v00, v10, v11), (v00, v11, v01)])
            else:
                elements.extend([(v00, v10, v01), (v10, v11, v01)])

    vertices = np.asarray(vertices, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    expected_triangles = 2 * NX * NY
    if len(elements) != expected_triangles:
        raise RuntimeError(
            f"Expected {expected_triangles} triangles, created {len(elements)}"
        )

    edge_nodes = ((0, 1), (1, 2), (2, 0))
    neighbors = np.full((len(elements), 3), -1, dtype=np.int64)
    neighbor_faces = np.full_like(neighbors, -1)
    unmatched = {}
    for element, triangle in enumerate(elements):
        for face, (a, b) in enumerate(edge_nodes):
            key = tuple(sorted((int(triangle[a]), int(triangle[b]))))
            if key in unmatched:
                other_element, other_face = unmatched.pop(key)
                neighbors[element, face] = other_element
                neighbor_faces[element, face] = other_face
                neighbors[other_element, other_face] = element
                neighbor_faces[other_element, other_face] = face
            else:
                unmatched[key] = (element, face)

    return TriangularMesh(vertices, elements, neighbors, neighbor_faces)


# =============================================================================
# 5. POINT LOCATION AND GPU SAMPLING
# =============================================================================

class PointSampler:
    def __init__(self, mesh, basis, invJ, origins, points, device, dtype):
        points = np.asarray(points, dtype=np.float64)
        triangulation = mtri.Triangulation(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements
        )
        finder = triangulation.get_trifinder()
        elements = np.asarray(finder(points[:, 0], points[:, 1]), dtype=np.int64)
        if np.any(elements < 0):
            raise ValueError(f"Sampling points outside mesh: {points[elements < 0][:5]}")
        offsets = points - origins[elements]
        reference = np.einsum(
            "pi,pji->pj", offsets, invJ[elements], optimize=True
        )
        phi = basis.evaluate(reference)
        self.elements = torch.as_tensor(elements, device=device, dtype=torch.long)
        self.phi = torch.as_tensor(phi, device=device, dtype=dtype)

    @torch.no_grad()
    def sample_gpu(self, coefficients):
        """Sample a small point set without transferring it to the CPU."""
        selected = coefficients[self.elements]
        return torch.einsum("pm,pmv->pv", self.phi, selected)

    @torch.no_grad()
    def sample(self, coefficients, batch_size=SAMPLING_BATCH_SIZE):
        pieces = []
        for start in range(0, len(self.elements), batch_size):
            stop = min(start + batch_size, len(self.elements))
            selected = coefficients[self.elements[start:stop]]
            values = torch.einsum(
                "pm,pmv->pv", self.phi[start:stop], selected
            )
            pieces.append(values.cpu())
        return torch.cat(pieces, dim=0).numpy()


# =============================================================================
# 6. VECTORIZED CUDA ADER-DG SOLVER
# =============================================================================

class GPUADERDG:
    edge_nodes = ((0, 1), (1, 2), (2, 0))

    def __init__(self, mesh, degree, device, dtype):
        self.mesh = mesh
        self.degree = int(degree)
        self.order = self.degree + 1
        self.device = device
        self.dtype = dtype
        self.nvars = 8
        self.basis = ModalBasis(degree)
        self.nmodes = self.basis.size

        ev = mesh.element_vertices
        self.origins_cpu = ev[:, 0]
        J = np.stack((ev[:, 1] - ev[:, 0], ev[:, 2] - ev[:, 0]), axis=2)
        detJ = np.linalg.det(J)
        if np.any(detJ <= 0.0):
            raise ValueError("Mesh contains non-counter-clockwise triangles")
        invJ = np.linalg.inv(J)
        self.invJ_cpu = invJ
        self.detJ_cpu = detJ

        centroids = np.mean(ev, axis=1)
        # Paper orientation: shale above y=700 m, sandstone below.
        material_ids = np.where(centroids[:, 1] >= INTERFACE_Y_M, 0, 1)
        self.material_ids_cpu = material_ids

        A_material = np.asarray([material.matrices()[0] for material in MATERIALS])
        B_material = np.asarray([material.matrices()[1] for material in MATERIALS])
        A = A_material[material_ids]
        B = B_material[material_ids]

        lengths = np.empty((len(ev), 3), dtype=np.float64)
        normals = np.empty((len(ev), 3, 2), dtype=np.float64)
        for face, (a, b) in enumerate(self.edge_nodes):
            tangent = ev[:, b] - ev[:, a]
            lengths[:, face] = np.linalg.norm(tangent, axis=1)
            normals[:, face, 0] = tangent[:, 1] / lengths[:, face]
            normals[:, face, 1] = -tangent[:, 0] / lengths[:, face]

        self.invJ = torch.as_tensor(invJ, device=device, dtype=dtype)
        self.detJ = torch.as_tensor(detJ, device=device, dtype=dtype)
        self.A = torch.as_tensor(A, device=device, dtype=dtype)
        self.B = torch.as_tensor(B, device=device, dtype=dtype)
        self.face_lengths = torch.as_tensor(lengths, device=device, dtype=dtype)
        normal_tensor = torch.as_tensor(normals, device=device, dtype=dtype)
        self.neighbors = torch.as_tensor(
            mesh.neighbors, device=device, dtype=torch.long
        )
        self.neighbor_faces = torch.as_tensor(
            mesh.neighbor_faces, device=device, dtype=torch.long
        )

        self.Dxi = torch.as_tensor(
            self.basis.d_xi, device=device, dtype=dtype
        )
        self.Deta = torch.as_tensor(
            self.basis.d_eta, device=device, dtype=dtype
        )
        self.Kxi = self.Dxi.T.contiguous()
        self.Keta = self.Deta.T.contiguous()

        nq = degree + 3
        ft, fw = line_quadrature(nq)
        reference_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        face_phi = []
        for a, b in self.edge_nodes:
            points = (
                reference_vertices[a][None, :]
                + ft[:, None]
                * (reference_vertices[b] - reference_vertices[a])[None, :]
            )
            face_phi.append(self.basis.evaluate(points))
        face_phi = np.asarray(face_phi)
        self.face_phi = torch.as_tensor(face_phi, device=device, dtype=dtype)
        self.face_phi_reversed = torch.flip(self.face_phi, dims=(1,))
        self.face_weights = torch.as_tensor(fw, device=device, dtype=dtype)

        # Isotropic maximum eigenvalue is independent of face direction.
        max_speeds = np.asarray([material.wave_speeds()[-1] for material in MATERIALS])
        alpha = torch.as_tensor(
            max_speeds[material_ids], device=device, dtype=dtype
        )[:, None, None, None]
        An = (
            normal_tensor[:, :, 0, None, None] * self.A[:, None, :, :]
            + normal_tensor[:, :, 1, None, None] * self.B[:, None, :, :]
        )
        identity = torch.eye(self.nvars, device=device, dtype=dtype)[None, None]
        self.face_plus = An + alpha * identity
        self.face_minus = An - alpha * identity

        self.source = self._build_point_source()
        self.cfl_diagnostic_s = self._cfl_diagnostic(lengths, max_speeds)

    def _cfl_diagnostic(self, lengths, max_speeds):
        diameter = 2.0 * self.detJ_cpu / np.sum(lengths, axis=1)
        return (
            0.40
            * np.min(diameter)
            / ((2 * self.degree + 1) * np.max(max_speeds))
        )

    def make_sampler(self, points):
        return PointSampler(
            self.mesh,
            self.basis,
            self.invJ_cpu,
            self.origins_cpu,
            points,
            self.device,
            self.dtype,
        )

    def _build_point_source(self):
        sampler = self.make_sampler(np.array([[SOURCE_X_M, SOURCE_Y_M]]))
        element = int(sampler.elements.item())
        phi = sampler.phi[0]
        delta_projection = phi / self.detJ[element]
        source = torch.zeros(
            (len(self.mesh.elements), self.nmodes, self.nvars),
            device=self.device,
            dtype=self.dtype,
        )
        source[element, :, 1] = POINT_SOURCE_STRENGTH * delta_projection
        source[element, :, 5] = -POINT_SOURCE_STRENGTH * delta_projection
        return source

    @torch.no_grad()
    def spatial_operator(self, coefficients):
        dxi = torch.einsum("mn,env->emv", self.Dxi, coefficients)
        deta = torch.einsum("mn,env->emv", self.Deta, coefficients)
        dx = (
            self.invJ[:, 0, 0, None, None] * dxi
            + self.invJ[:, 1, 0, None, None] * deta
        )
        dy = (
            self.invJ[:, 0, 1, None, None] * dxi
            + self.invJ[:, 1, 1, None, None] * deta
        )
        return -(
            torch.bmm(dx, self.A.transpose(1, 2))
            + torch.bmm(dy, self.B.transpose(1, 2))
        )

    @torch.no_grad()
    def predictor_integral(self, coefficients, dt):
        integral = dt * coefficients
        term = coefficients
        factor = dt
        for k in range(1, self.order):
            term = self.spatial_operator(term)
            factor *= dt / (k + 1)
            integral = integral + factor * term
        return integral

    @torch.no_grad()
    def step(self, coefficients, dt, source_time_integral):
        time_integral = self.predictor_integral(coefficients, dt)

        weak_xi = torch.einsum("mn,env->emv", self.Kxi, time_integral)
        weak_eta = torch.einsum("mn,env->emv", self.Keta, time_integral)
        weak_x = (
            self.invJ[:, 0, 0, None, None] * weak_xi
            + self.invJ[:, 1, 0, None, None] * weak_eta
        )
        weak_y = (
            self.invJ[:, 0, 1, None, None] * weak_xi
            + self.invJ[:, 1, 1, None, None] * weak_eta
        )
        volume = (
            torch.bmm(weak_x, self.A.transpose(1, 2))
            + torch.bmm(weak_y, self.B.transpose(1, 2))
        )

        surface = torch.zeros_like(coefficients)
        for face in range(3):
            phi_left = self.face_phi[face]
            qleft = torch.einsum("qm,emv->eqv", phi_left, time_integral)
            outgoing = torch.bmm(
                qleft, self.face_plus[:, face].transpose(1, 2)
            )

            neighbor = self.neighbors[:, face]
            interior = neighbor >= 0
            safe_neighbor = torch.clamp(neighbor, min=0)
            safe_neighbor_face = torch.clamp(
                self.neighbor_faces[:, face], min=0
            )
            phi_right = self.face_phi_reversed[safe_neighbor_face]
            qright = torch.bmm(phi_right, time_integral[safe_neighbor])
            incoming = torch.bmm(
                qright, self.face_minus[:, face].transpose(1, 2)
            )
            incoming = incoming * interior[:, None, None]
            flux = 0.5 * (outgoing + incoming)

            weighted_flux = self.face_weights[None, :, None] * flux
            face_contribution = torch.einsum(
                "qm,eqv->emv", phi_left, weighted_flux
            )
            scale = (
                self.face_lengths[:, face] / self.detJ
            )[:, None, None]
            surface = surface + scale * face_contribution

        return (
            coefficients
            + volume
            - surface
            + source_time_integral * self.source
        )


# =============================================================================
# 7. TIME SCHEDULE, RICKER SOURCE, AND CHECKPOINTING
# =============================================================================

def ricker(t):
    a = np.pi * SOURCE_FREQUENCY_HZ * (t - SOURCE_DELAY_S)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)


def integrated_ricker(t, dt, n=6):
    x, w = np.polynomial.legendre.leggauss(n)
    times = t + 0.5 * dt * (x + 1.0)
    return float(0.5 * dt * np.sum(w * ricker(times)))


def build_time_schedule():
    snapshots = sorted(float(value) for value in SNAPSHOT_TIMES_S)
    steps, end_times, source_integrals = [], [], []
    t, next_snapshot = 0.0, 0
    while t < FINAL_TIME_S - 10.0 * np.finfo(float).eps:
        dt = min(PAPER_DT_S, FINAL_TIME_S - t)
        if (
            next_snapshot < len(snapshots)
            and t < snapshots[next_snapshot] < t + dt
        ):
            dt = snapshots[next_snapshot] - t
        source_integrals.append(integrated_ricker(t, dt))
        steps.append(dt)
        t += dt
        end_times.append(t)
        if (
            next_snapshot < len(snapshots)
            and abs(t - snapshots[next_snapshot]) < 1.0e-11
        ):
            next_snapshot += 1
    return np.asarray(steps), np.asarray(end_times), np.asarray(source_integrals)


def save_checkpoint(path, completed_steps, coefficients, receiver_history,
                    snapshots, snapshot_times):
    payload = {
        "completed_steps": int(completed_steps),
        "coefficients": coefficients.detach().cpu(),
        "receiver_history": np.asarray(receiver_history),
        "snapshots": np.asarray(snapshots),
        "snapshot_times": np.asarray(snapshot_times),
    }
    torch.save(payload, path)


def load_checkpoint(path, device, dtype):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coefficients = payload["coefficients"].to(device=device, dtype=dtype)
    receiver_history = np.asarray(payload["receiver_history"])
    snapshots = list(np.asarray(payload["snapshots"]))
    snapshot_times = list(np.asarray(payload["snapshot_times"]))
    return (
        int(payload["completed_steps"]),
        coefficients,
        receiver_history,
        snapshots,
        snapshot_times,
    )


# =============================================================================
# 8. SIMULATION AND OUTPUT
# =============================================================================

def material_grid(x, y):
    X, Y = np.meshgrid(x, y)
    shale = Y >= INTERFACE_Y_M
    fields = np.empty((len(y), len(x), 10), dtype=np.float32)
    names = np.array(
        [
            "rho_s", "rho_f", "lambda_matrix", "mu_matrix", "bulk_solid",
            "bulk_fluid", "porosity", "log10_permeability", "tortuosity",
            "material_id",
        ]
    )
    for mask, material, material_id in (
        (shale, SHALE, 0.0), (~shale, SANDSTONE, 1.0)
    ):
        fields[mask, 0] = material.rho_s
        fields[mask, 1] = material.rho_f
        fields[mask, 2] = material.lambda_matrix
        fields[mask, 3] = material.mu_matrix
        fields[mask, 4] = material.bulk_solid
        fields[mask, 5] = material.bulk_fluid
        fields[mask, 6] = material.porosity
        fields[mask, 7] = np.log10(material.permeability)
        fields[mask, 8] = material.tortuosity
        fields[mask, 9] = material_id
    return X, Y, fields, names


def run_simulation():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Colab select Runtime -> Change runtime type -> GPU."
        )
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / CHECKPOINT_FILE

    print("Building 132,014-element irregular mesh on CPU...")
    mesh = build_paper_mesh()
    print("Precomputing O5 operators and transferring them to CUDA...")
    solver = GPUADERDG(mesh, POLYNOMIAL_DEGREE, device, TORCH_DTYPE)

    x = np.linspace(DOMAIN_X_M[0], DOMAIN_X_M[1], OUTPUT_NX)
    y = np.linspace(DOMAIN_Y_M[0], DOMAIN_Y_M[1], OUTPUT_NY)
    X, Y = np.meshgrid(x, y)
    uniform_points = np.column_stack((X.ravel(), Y.ravel()))
    output_sampler = solver.make_sampler(uniform_points)
    receiver_sampler = solver.make_sampler(RECEIVERS_M)

    dt_steps, end_times, source_integrals = build_time_schedule()
    total_steps = len(dt_steps)

    print("\nPAPER-SCALE GPU ADER-DG")
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"dtype                : {TORCH_DTYPE}")
    print(f"triangles            : {len(mesh.elements):,}")
    print(f"modal functions      : {solver.nmodes}")
    print(f"state unknowns       : {len(mesh.elements)*solver.nmodes*8:,}")
    print(f"ADER-DG              : O{solver.order}")
    print(f"paper dt             : {PAPER_DT_S:.6e} s")
    print(f"CFL diagnostic dt    : {solver.cfl_diagnostic_s:.6e} s")
    print(f"time steps           : {total_steps:,}")
    print("upper/lower layers   : shale / sandstone")
    print(f"shale wave speeds    : {SHALE.wave_speeds()}")
    print(f"sandstone wave speeds: {SANDSTONE.wave_speeds()}")
    if PAPER_DT_S > solver.cfl_diagnostic_s:
        print("WARNING: published dt exceeds the conservative CFL diagnostic for")
        print("this reconstructed mesh. Monitor stability and reduce PAPER_DT_S if needed.")

    if RESUME_IF_CHECKPOINT_EXISTS and checkpoint_path.exists():
        (
            start_step,
            coefficients,
            receiver_history,
            snapshots,
            snapshot_times,
        ) = load_checkpoint(checkpoint_path, device, TORCH_DTYPE)
        print(f"Resuming from checkpoint after step {start_step:,}.")
    else:
        start_step = 0
        coefficients = torch.zeros(
            (len(mesh.elements), solver.nmodes, solver.nvars),
            device=device,
            dtype=TORCH_DTYPE,
        )
        receiver_history = np.empty((1, len(RECEIVERS_M), solver.nvars), dtype=np.float32)
        receiver_history[0] = receiver_sampler.sample(coefficients)
        snapshots, snapshot_times = [], []

    receiver_history_gpu = torch.empty(
        (total_steps + 1, len(RECEIVERS_M), solver.nvars),
        device=device,
        dtype=TORCH_DTYPE,
    )
    receiver_history_gpu[:start_step + 1] = torch.as_tensor(
        receiver_history, device=device, dtype=TORCH_DTYPE
    )

    next_snapshot = len(snapshot_times)
    requested_snapshots = sorted(SNAPSHOT_TIMES_S)
    started = time.perf_counter()

    for step_index in range(start_step, total_steps):
        coefficients = solver.step(
            coefficients,
            float(dt_steps[step_index]),
            float(source_integrals[step_index]),
        )
        completed = step_index + 1
        receiver_history_gpu[completed] = receiver_sampler.sample_gpu(coefficients)
        current_time = float(end_times[step_index])

        if (
            next_snapshot < len(requested_snapshots)
            and abs(current_time - requested_snapshots[next_snapshot]) < 1.0e-10
        ):
            field = output_sampler.sample(coefficients)
            snapshots.append(field.reshape(OUTPUT_NY, OUTPUT_NX, solver.nvars))
            snapshot_times.append(current_time)
            print(f"Saved snapshot at t={current_time:.5f} s")
            next_snapshot += 1

        if completed % PROGRESS_EVERY_STEPS == 0 or completed == total_steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            run_steps = completed - start_step
            seconds_per_step = elapsed / max(run_steps, 1)
            remaining = seconds_per_step * (total_steps - completed)
            maximum = float(torch.max(torch.abs(coefficients)).item())
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"step {completed:6d}/{total_steps}: t={current_time:.5f} s, "
                f"|Q|max={maximum:.4e}, GPU={memory_gb:.2f} GB, "
                f"ETA={remaining/60.0:.1f} min"
            )

        if (
            CHECKPOINT_EVERY_STEPS > 0
            and completed % CHECKPOINT_EVERY_STEPS == 0
        ):
            save_checkpoint(
                checkpoint_path,
                completed,
                coefficients,
                receiver_history_gpu[:completed + 1].cpu().numpy(),
                snapshots,
                snapshot_times,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    receiver_history = receiver_history_gpu[:total_steps + 1].cpu().numpy()
    snapshots = np.asarray(snapshots, dtype=np.float32)
    times = np.concatenate(([0.0], end_times[:len(receiver_history) - 1]))
    _, _, material_fields, material_names = material_grid(x, y)

    output_path = OUTPUT_DIR / OUTPUT_FILE
    np.savez_compressed(
        output_path,
        x_m=x,
        y_m=y,
        times_s=times,
        snapshot_times_s=np.asarray(snapshot_times),
        wavefields=snapshots,
        state_names=np.array(
            ["sxx", "syy", "sxy", "vx", "vy", "p", "qx", "qy"]
        ),
        material_fields=material_fields,
        material_names=material_names,
        receivers_m=RECEIVERS_M,
        receiver_history=receiver_history,
        source_location_m=np.array([SOURCE_X_M, SOURCE_Y_M]),
        source_frequency_hz=SOURCE_FREQUENCY_HZ,
        source_strength=POINT_SOURCE_STRENGTH,
        interface_y_m=INTERFACE_Y_M,
        triangles=len(mesh.elements),
        polynomial_degree=POLYNOMIAL_DEGREE,
        nominal_dt_s=PAPER_DT_S,
    )
    save_checkpoint(
        checkpoint_path,
        total_steps,
        coefficients,
        receiver_history,
        snapshots,
        snapshot_times,
    )
    print(f"Simulation runtime: {runtime/60.0:.2f} min")
    print(f"Saved output: {output_path}")
    return (
        x,
        y,
        snapshots,
        np.asarray(snapshot_times),
        times,
        receiver_history,
        material_fields,
    )


# =============================================================================
# 9. PAPER-COMPARISON FIGURES
# =============================================================================

def plot_results(x, y, snapshots, snapshot_times, times, receivers, materials):
    saved = []
    material_cmap = ListedColormap(["#59636f", "#d7a348"])
    material_norm = BoundaryNorm([-0.5, 0.5, 1.5], material_cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    image = ax.pcolormesh(
        x, y, materials[:, :, 9], shading="auto",
        cmap=material_cmap, norm=material_norm,
    )
    ax.axhline(INTERFACE_Y_M, color="white", lw=2.0, ls="--")
    ax.scatter(SOURCE_X_M, SOURCE_Y_M, marker="*", s=210, c="#d62728",
               edgecolor="white", linewidth=0.8, label="50-Hz source")
    ax.scatter(RECEIVERS_M[:, 0], RECEIVERS_M[:, 1], marker="v", s=70,
               c="black", edgecolor="white", linewidth=0.5, label="receivers")
    for i, (rx, ry) in enumerate(RECEIVERS_M, start=1):
        ax.annotate(f"R{i}", (rx, ry), xytext=(8, 6), textcoords="offset points")
    ax.text(55, 1180, "SHALE", color="white", fontsize=13, weight="bold")
    ax.text(55, 240, "SANDSTONE", color="#3b2a0d", fontsize=13, weight="bold")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Paper two-layer configuration")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.85)
    cbar.ax.set_yticklabels(["Shale", "Sandstone"])
    path = OUTPUT_DIR / "01_paper_model_setup.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    saved.append(path)
    plt.show()

    snapshot_index = int(np.argmin(np.abs(snapshot_times - 0.25)))
    field = snapshots[snapshot_index, :, :, 4]
    limit = max(np.percentile(np.abs(field), 99.5), 1.0e-30)
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    image = ax.pcolormesh(
        x, y, field, shading="auto", cmap="seismic", vmin=-limit, vmax=limit
    )
    ax.axhline(INTERFACE_Y_M, color="black", lw=0.8, ls="--")
    ax.scatter(SOURCE_X_M, SOURCE_Y_M, marker="o", s=30, c="black")
    ax.scatter(RECEIVERS_M[:, 0], RECEIVERS_M[:, 1], marker="o", s=25,
               facecolors="none", edgecolors="black")
    ax.set(
        xlabel="x (m)", ylabel="y (m)",
        title=f"Solid vertical velocity $v_y$ at t={snapshot_times[snapshot_index]:.2f} s",
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(image, ax=ax, label="$v_y$ (m/s)")
    path = OUTPUT_DIR / "02_figure_9_vertical_velocity.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    saved.append(path)
    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True,
                             constrained_layout=True)
    for receiver_index in range(3):
        for column, (component, name) in enumerate(((3, "$v_x$"), (4, "$v_y$"))):
            ax = axes[receiver_index, column]
            trace = receivers[:, receiver_index, component]
            ax.plot(times, trace, color="#155a8a", lw=1.1)
            ax.axhline(0.0, color="black", lw=0.5)
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"R{receiver_index + 1} {name}\n(m/s)")
    axes[0, 0].set_title("Horizontal solid velocity")
    axes[0, 1].set_title("Vertical solid velocity")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Receiver seismograms for comparison with paper Figure 10",
                 fontsize=15, weight="bold")
    path = OUTPUT_DIR / "03_figure_10_receiver_seismograms.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    saved.append(path)
    plt.show()

    final = snapshots[-1]
    panels = [
        (final[:, :, 5] / 1.0e6, "Pore pressure", "p (MPa)"),
        (final[:, :, 4], "Solid vertical velocity", "$v_y$ (m/s)"),
        (final[:, :, 7], "Relative-fluid vertical velocity", "$q_y$ (m/s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for ax, (panel, title, label) in zip(axes, panels):
        bound = max(np.percentile(np.abs(panel), 99.5), 1.0e-30)
        image = ax.pcolormesh(
            x, y, panel, shading="auto", cmap="seismic", vmin=-bound, vmax=bound
        )
        ax.axhline(INTERFACE_Y_M, color="black", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=ax, label=label, shrink=0.80)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"Coupled response at t={snapshot_times[-1]:.2f} s",
                 fontsize=15, weight="bold")
    path = OUTPUT_DIR / "04_final_coupled_response.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    saved.append(path)
    plt.show()

    print("Saved figures:")
    for path in saved:
        print(f"  {path}")
    return saved


def main():
    results = run_simulation()
    figures = plot_results(*results)
    return results, figures


if __name__ == "__main__":
    gpu_results, gpu_figures = main()
