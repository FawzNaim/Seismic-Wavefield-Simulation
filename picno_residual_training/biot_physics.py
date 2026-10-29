"""Differentiable discrete inviscid/linear Biot residuals from saved A, B, E.

Bulk residual: U_t + A U_x + B U_y - E U - S = 0.
BDF2 time difference, fourth-order centered interior spatial differences.
Every history state is true; only the next state is predicted.
The continuum residual on an interpolated DG grid is not the DG update.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def dilate(mask, radius):
    if radius == 0:
        return mask
    return F.max_pool2d(mask.float()[None, None], 2*radius+1,
                        stride=1, padding=radius)[0, 0].bool()


def stencil_mask(layer_ids, projected_source, boundary_cells=4,
                 interface_cells=3, source_cells=4):
    """Keep fourth-order stencils within a material and away from the DG source."""
    if boundary_cells < 2 or interface_cells < 2 or source_cells < 2:
        raise ValueError('Mask margins must be >=2 for the fourth-order stencil')
    ids = torch.as_tensor(layer_ids)
    interface = torch.zeros_like(ids, dtype=torch.bool)
    change = ids[:, 1:] != ids[:, :-1]
    interface[:, 1:] |= change
    interface[:, :-1] |= change
    change = ids[1:] != ids[:-1]
    interface[1:] |= change
    interface[:-1] |= change
    source = torch.as_tensor(projected_source).abs().amax(dim=0)
    source = source > 0  # Entire support of sampled DG projection, not Gaussian map
    valid = ~(dilate(interface, interface_cells) | dilate(source, source_cells))
    valid[:boundary_cells] = False
    valid[-boundary_cells:] = False
    valid[:, :boundary_cells] = False
    valid[:, -boundary_cells:] = False
    if not valid.any():
        raise ValueError('No valid bulk residual cells; reduce margins or use finer grid')
    return valid[2:-2, 2:-2].float()


def spatial_derivatives(u, dx, dy):
    """Return derivatives at [2:-2,2:-2] cell centers, physical spacings."""
    ux = (-u[..., 2:-2, 4:] + 8*u[..., 2:-2, 3:-1]
          -8*u[..., 2:-2, 1:-3] + u[..., 2:-2, :-4]) / (12*dx)
    uy = (-u[..., 4:, 2:-2] + 8*u[..., 3:-1, 2:-2]
          -8*u[..., 1:-3, 2:-2] + u[..., :-4, 2:-2]) / (12*dy)
    return ux, uy


def apply_layer_matrices(matrices, field, ids):
    # matrices [batch,layer,out,in], field [batch,in,y,x]
    terms = torch.einsum('blci,biyx->blcyx', matrices, field)
    mask = F.one_hot(ids.long(), num_classes=2).permute(2, 0, 1)
    return (terms * mask[None, :, None]).sum(dim=1)


def incoming_matrices(matrices, scales):
    """Incoming characteristic flux in normalized field coordinates.

    Uses A_n^- / max_wave_speed. Nondimensionalization precedes eigendecomposition.
    A_n^- U=0 expresses zero incoming characteristics; it is approximate absorption.
    """
    scaled = matrices * scales[None, :]/scales[:, None]
    output = []
    for m in scaled:
        values, vectors = np.linalg.eig(m)
        if np.max(np.abs(values.imag)) > 1e-5:
            raise ValueError('Non-real characteristic speeds in boundary matrix')
        values = values.real
        negative = np.minimum(values, 0)
        speed = max(float(np.max(np.abs(values))), 1e-12)
        minus = vectors @ np.diag(negative) @ np.linalg.inv(vectors) / speed
        if np.max(np.abs(minus.imag)) > 1e-5:
            raise ValueError('Complex incoming characteristic matrix')
        output.append(minus.real)
    return np.array(output, dtype=np.float32)


class BiotResidual(nn.Module):
    def __init__(self, meta, statistics, args):
        super().__init__()
        self.dt = float(meta['dt'])
        self.dx, self.dy = float(meta['dx']), float(meta['dy'])
        self.time_scale = float(args.physics_time_scale)
        if self.time_scale <= 0:
            raise ValueError('physics-time-scale must be positive')
        self.register_buffer('mean', torch.tensor(statistics['output_mean'], dtype=torch.float32)[None, :, None, None])
        self.register_buffer('std', torch.tensor(statistics['output_std'], dtype=torch.float32)[None, :, None, None])
        self.register_buffer('ids', torch.tensor(meta['layer_ids'], dtype=torch.long))
        self.register_buffer('source', torch.tensor(meta['projected_source'], dtype=torch.float32)[None])
        self.register_buffer('valid', stencil_mask(meta['layer_ids'], meta['projected_source'],
                             args.boundary_cells, args.interface_cells, args.source_cells))
        self.frequency = float(meta['source_frequency_hz'])
        self.delay = float(meta['source_delay_s'])
        y = np.asarray(meta['y'])
        interface = float(meta['interface_y_m'])
        above = np.flatnonzero(y >= interface)[:2]
        below = np.flatnonzero(y < interface)[-2:]
        if len(above) < 2 or len(below) < 2:
            raise ValueError('Two points per side needed for interface extrapolation')
        self.trace_indices = [above.tolist(), below.tolist()]
        self.trace_weights = []
        for indices in self.trace_indices:
            a, b = y[indices]
            self.trace_weights.append(((b-interface)/(b-a), (interface-a)/(b-a)))
        self.edge_margin = args.boundary_cells

    def physical(self, normalized):
        return normalized.float()*self.std + self.mean

    def residual(self, predicted, previous, older, A, B, E, target_times):
        # All operations outside network autocast, in float32 physical units.
        u, prev, old = [self.physical(x) for x in (predicted, previous, older)]
        ut = (3*u - 4*prev + old)/(2*self.dt)
        ux, uy = spatial_derivatives(u, self.dx, self.dy)
        ids = self.ids[2:-2, 2:-2]
        center = u[..., 2:-2, 2:-2]
        a = torch.pi*self.frequency*(target_times.float()-self.delay)
        ricker = (1-2*a.square())*torch.exp(-a.square())
        s = self.source[..., 2:-2, 2:-2]*ricker[:, None, None, None]
        r = ut[..., 2:-2, 2:-2] + apply_layer_matrices(A.float(), ux, ids)
        r = r + apply_layer_matrices(B.float(), uy, ids)
        r = r - apply_layer_matrices(E.float(), center, ids) - s
        return r*(self.time_scale/self.std)

    def bulk_loss(self, residual):
        return (residual.square()*self.valid[None, None]).sum() / (
            residual.shape[0]*residual.shape[1]*self.valid.sum())

    def interface_loss(self, predicted):
        # Perfectly bonded, open-pore interface, linear one-sided trace estimates.
        u = self.physical(predicted)/self.std
        traces = []
        for indices, weights in zip(self.trace_indices, self.trace_weights):
            traces.append(weights[0]*u[..., indices[0], :] + weights[1]*u[..., indices[1], :])
        channels = [1, 2, 3, 4, 5, 7]  # syy,sxy,vx,vy,p,qy; sxx and qx may jump
        e = self.edge_margin
        return (traces[0][:, channels, e:-e]-traces[1][:, channels, e:-e]).square().mean()

    def boundary_loss(self, predicted, incoming):
        # incoming [batch,edge,layer,8,8], edges left/right/bottom/top.
        u = self.physical(predicted)/self.std
        e = self.edge_margin
        traces = (u[..., e:-e, 0], u[..., e:-e, -1],
                  u[..., 0, e:-e], u[..., -1, e:-e])
        ids = (self.ids[e:-e, 0], self.ids[e:-e, -1],
               self.ids[0, e:-e], self.ids[-1, e:-e])
        loss = u.new_zeros(())
        for k, (trace, labels) in enumerate(zip(traces, ids)):
            flux = torch.einsum('blci,bip->blcp', incoming[:, k].float(), trace)
            select = F.one_hot(labels, 2).T[None, :, None]
            loss = loss+(flux*select).sum(dim=1).square().mean()/4
        return loss
