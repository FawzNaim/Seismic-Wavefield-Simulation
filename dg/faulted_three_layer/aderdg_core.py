"""ADER-DG numerical core adapted from the previous two-layer solver.
Three-material classification and face wave-speed bound generalized.
This is a numerical model, not a verified reproduction of a published case.
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


DOMAIN_X_M = (0.0, 1500.0)


DOMAIN_Y_M = (0.0, 1400.0)


SOURCE_X_M = 250.0


SOURCE_Y_M = 200.0


SOURCE_FREQUENCY_HZ = 50.0


SOURCE_DELAY_S = 0.04


POINT_SOURCE_STRENGTH = 2.0e10


SAMPLING_BATCH_SIZE = 65536


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


@dataclass
class TriangularMesh:
    vertices: np.ndarray
    elements: np.ndarray
    neighbors: np.ndarray
    neighbor_faces: np.ndarray
    material_ids: np.ndarray

    @property
    def element_vertices(self):
        return self.vertices[self.elements]


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
        # Conforming fault/layer mesh stores one material ID per element.
        material_ids = mesh.material_ids.copy()
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
        # Use the larger adjacent speed on both sides of an interior face.
        neighbor_ids = material_ids[np.maximum(mesh.neighbors, 0)]
        face_speeds = np.maximum(max_speeds[material_ids, None], max_speeds[neighbor_ids])
        face_speeds[mesh.neighbors < 0] = np.broadcast_to(max_speeds[material_ids, None], face_speeds.shape)[mesh.neighbors < 0]
        alpha = torch.as_tensor(face_speeds, device=device, dtype=dtype)[:, :, None, None]
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


def ricker(t):
    a = np.pi * SOURCE_FREQUENCY_HZ * (t - SOURCE_DELAY_S)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)


def integrated_ricker(t, dt, n=6):
    x, w = np.polynomial.legendre.leggauss(n)
    times = t + 0.5 * dt * (x + 1.0)
    return float(0.5 * dt * np.sum(w * ricker(times)))


