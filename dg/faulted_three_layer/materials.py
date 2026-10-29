"""Admissible end members and a documented 30/70 modeling approximation."""
from dataclasses import asdict
import numpy as np
import aderdg_core as core

BASE_SHALE,BASE_SANDSTONE=core.SHALE,core.SANDSTONE
INPUT_NAMES=('rho_s','rho_f','lambda_matrix','mu_matrix','bulk_solid','bulk_fluid',
             'porosity','log10_permeability','tortuosity','material_id')
MATERIAL_NAMES=('shale','shaly_sandstone','sandstone')


def hill(a,b,f=.3):
    return .5*(f*a+(1-f)*b+1/(f/a+(1-f)/b))


def mixed(shale,sandstone):
    f=.3
    K=hill(shale.bulk_matrix,sandstone.bulk_matrix,f)
    mu=hill(shale.mu_matrix,sandstone.mu_matrix,f)
    if not np.isclose(shale.rho_f,sandstone.rho_f) or not np.isclose(shale.bulk_fluid,sandstone.bulk_fluid):
        raise ValueError('Both end members must share the same pore fluid')
    return core.BiotMaterial('shaly_sandstone',f*shale.rho_s+(1-f)*sandstone.rho_s,
        shale.rho_f,K-2*mu/3,mu,hill(shale.bulk_solid,sandstone.bulk_solid,f),shale.bulk_fluid,
        f*shale.porosity+(1-f)*sandstone.porosity,
        float(np.exp(f*np.log(shale.permeability)+(1-f)*np.log(sandstone.permeability))),
        f*shale.tortuosity+(1-f)*sandstone.tortuosity,0.)


def valid(m):
    try:
        eigen=np.linalg.eigvals(m.matrices()[0])
        return (0.01<m.porosity<.6 and m.bulk_matrix<m.bulk_solid and m.mass_determinant>0
                and m.coupling_M>0 and abs(eigen.imag).max()<1e-5 and len(m.wave_speeds())==3
                and np.isfinite(eigen).all())
    except (ValueError,np.linalg.LinAlgError):return False


def sample(seed,split_code,index):
    rng=np.random.default_rng(np.random.SeedSequence([seed,split_code,index]))
    fluid_rho=BASE_SHALE.rho_f*rng.uniform(.98,1.02)
    fluid_K=BASE_SHALE.bulk_fluid*rng.uniform(.9,1.1)
    def draw(reference):
        d=asdict(reference)
        for name,fraction in (('rho_s',.05),('lambda_matrix',.10),('mu_matrix',.10),
                              ('bulk_solid',.10),('tortuosity',.10)):
            d[name]*=rng.uniform(1-fraction,1+fraction)
        d.update(rho_f=fluid_rho,bulk_fluid=fluid_K,porosity=d['porosity']+rng.uniform(-.02,.02),viscosity=0.)
        return core.BiotMaterial(**d)
    for _ in range(1000):
        shale,sand=draw(BASE_SHALE),draw(BASE_SANDSTONE)
        mix=mixed(shale,sand)
        if all(valid(m) for m in (shale,mix,sand)):return shale,mix,sand
    raise RuntimeError('Could not sample an admissible three-material realization')


def vector(m):
    return np.array([m.rho_s,m.rho_f,m.lambda_matrix,m.mu_matrix,m.bulk_solid,m.bulk_fluid,
                     m.porosity,np.log10(m.permeability),m.tortuosity])


def fields(ids,materials):
    values=np.stack([np.r_[vector(m),i] for i,m in enumerate(materials)])
    return np.moveaxis(values[ids],-1,0).astype(np.float32)
