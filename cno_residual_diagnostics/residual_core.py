"""Eight-equation Biot bulk residuals on the saved uniform grid.

No gradients or training. Residuals use the original training operator:
BDF2 in saved-frame time, fourth-order centered space, saved A/B/E and source.
"""
import numpy as np
import torch
from biot_physics import spatial_derivatives, apply_layer_matrices


def physical_terms(current, previous, older, A, B, E, ids, source, times,
                   dt, dx, dz, frequency, delay):
    """Return [batch,5,8,z-4,x-4] signed PDE terms in physical units/s."""
    ux, uz = spatial_derivatives(current, dx, dz)
    center = current[..., 2:-2, 2:-2]
    layer_ids = ids[2:-2, 2:-2]
    ut = ((3*current-4*previous+older)/(2*dt))[..., 2:-2, 2:-2]
    phase = torch.pi*frequency*(times-delay)
    wavelet = (1-2*phase.square())*torch.exp(-phase.square())
    s = source[..., 2:-2, 2:-2]*wavelet[:, None, None, None]
    return torch.stack((ut, apply_layer_matrices(A,ux,layer_ids),
                        apply_layer_matrices(B,uz,layer_ids),
                        -apply_layer_matrices(E,center,layer_ids), -s),dim=1)


class ResidualComparison:
    """Accumulate squared/absolute/signed residuals; do not average signed R only."""
    def __init__(self, nt, valid, scale, time_scale):
        self.valid = valid.bool()
        self.pixels = int(valid.sum())
        if not self.pixels:
            raise ValueError('Residual mask is empty')
        self.factor = time_scale/np.asarray(scale,np.float64)
        self.unscale = np.asarray(scale,np.float64)/time_scale
        self.counts = np.zeros(nt,np.int64)
        self.sq = {k:np.zeros((nt,8)) for k in ('cno','reference','difference')}
        self.absolute = {k:np.zeros(8) for k in ('cno','reference')}
        self.signed = {k:np.zeros(8) for k in ('cno','reference')}
        self.maximum = {k:np.zeros(8) for k in ('cno','reference')}
        self.term_sq = {k:np.zeros((5,8)) for k in ('cno','reference')}

    def update(self, indices, cno_terms, reference_terms):
        factor = torch.tensor(self.factor,device=cno_terms.device,dtype=cno_terms.dtype)
        selected = {k:v[..., self.valid]*factor[None,None,:,None]
                    for k,v in (('cno',cno_terms),('reference',reference_terms))}
        residuals = {k:v.sum(1) for k,v in selected.items()}
        for key,r in residuals.items():
            r = r.double()
            np.add.at(self.sq[key],indices,r.square().sum(-1).cpu().numpy())
            self.absolute[key] += r.abs().sum((0,2)).cpu().numpy()
            self.signed[key] += r.sum((0,2)).cpu().numpy()
            self.maximum[key] = np.maximum(self.maximum[key],r.abs().amax((0,2)).cpu().numpy())
            self.term_sq[key] += selected[key].double().square().sum((0,3)).cpu().numpy()
        delta = (residuals['cno']-residuals['reference']).double()
        np.add.at(self.sq['difference'],indices,delta.square().sum(-1).cpu().numpy())
        np.add.at(self.counts,indices,self.pixels)

    def summary(self,names):
        count = int(self.counts.sum())
        if not count:
            raise ValueError('No residual examples were evaluated')
        rms = {k:np.sqrt(v.sum(0)/count) for k,v in self.sq.items()}
        result = {}
        for c,name in enumerate(names):
            row = {}
            for key in ('cno','reference'):
                denom = np.sqrt(self.term_sq[key][:,c]/count).sum()
                row[key] = dict(rms_dimensionless=float(rms[key][c]),
                    mae_dimensionless=float(self.absolute[key][c]/count),
                    signed_mean_dimensionless=float(self.signed[key][c]/count),
                    maximum_absolute_dimensionless=float(self.maximum[key][c]),
                    rms_physical=float(rms[key][c]*self.unscale[c]),
                    relative_balance_rms=float(rms[key][c]/denom) if denom>0 else None)
            row['difference_rms_dimensionless'] = float(rms['difference'][c])
            row['cno_to_reference_rms_ratio'] = float(rms['cno'][c]/rms['reference'][c]) if rms['reference'][c]>0 else None
            row['difference_to_reference_rms_ratio'] = float(rms['difference'][c]/rms['reference'][c]) if rms['reference'][c]>0 else None
            result[name] = row
        return dict(valid_pixel_equation_samples=count,by_channel=result)
