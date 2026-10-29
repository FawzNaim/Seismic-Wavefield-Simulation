"""CPU geometry, material, interface and DG-consistency checks."""
import numpy as np
import torch
import aderdg_core as core
import fault_geometry as geo
import materials as rock
import interface_flux
from generate_faulted_biot_dataset import source_integral


def main():
    torch.set_num_threads(2)
    np.testing.assert_array_equal(geo.depth_to_solver(geo.SOURCE_DEPTH_M),[250,200])
    np.testing.assert_array_equal(geo.depth_to_solver(geo.RECEIVERS_DEPTH_M),[[1000,400],[1000,800],[1000,1200]])
    assert sum(f.direction==1 for f in geo.FAULTS)==3 and sum(f.direction==-1 for f in geo.FAULTS)==2
    x=np.linspace(0,1500,101)
    assert np.all(geo.material_ids_depth(x,0)==0) and np.all(geo.material_ids_depth(x,1400)==2)
    assert np.all(geo.throw_at_depth([0,200,950,1400])<1e-10)
    for j in range(30):
        mats=rock.sample(20260925,0,j)
        assert all(rock.valid(m) for m in mats)
        assert np.isclose(mats[1].porosity,.3*mats[0].porosity+.7*mats[2].porosity)
    mats=rock.sample(20260925,0,0)
    rng=np.random.default_rng(12)
    for angle in np.linspace(0,2*np.pi,17):
        n=np.array([np.cos(angle),np.sin(angle)])
        flux,tr=interface_flux.coupled_flux(mats[0],mats[2],n)
        UL,UR=rng.normal(size=(2,8))
        left=tr[0]@UL+tr[1]@UR;right=tr[2]@UL+tr[3]@UR
        assert np.max(np.abs(interface_flux.continuity_matrix(n)@(left-right)))<1e-10
        A,B=mats[0].matrices()
        np.testing.assert_allclose(flux[0]+flux[1],n[0]*A+n[1]*B,rtol=1e-10,atol=1e-5)
    for t in (0.,.02,.04,.10):
        np.testing.assert_allclose(source_integral(t,1e-4),core.integrated_ricker(t,1e-4),rtol=1e-10,atol=1e-15)
    mesh=geo.build_mesh(4000.)
    core.MATERIALS=mats
    solver=core.GPUADERDG(mesh,2,torch.device('cpu'),torch.float64)
    count=interface_flux.install(solver,mats)
    assert count>0
    zero=torch.zeros(len(mesh.elements),solver.nmodes,8,dtype=torch.float64)
    assert torch.equal(solver.step(zero,solver.cfl_diagnostic_s*.5,0.),zero)
    # Constant common physical state must be preserved away from outer boundaries.
    q=zero.clone();state=np.array([1e4,2e4,3e3,.001,.002,1e4,.0001,.0002])
    q[:,0]=torch.tensor(state/np.sqrt(2.))
    result=solver.step(q,solver.cfl_diagnostic_s*.5,0.)
    interior=(mesh.neighbors>=0).all(1)
    scale=torch.tensor([1e4]*3+[.001,.001,1e4,.0001,.0001])
    error=float(((result-q)[interior]/scale).abs().max())
    assert error<1e-8,error
    print(f'PASS: coordinate conversion, finite fault tips/dips, 30 material draws, interface conditions, Ricker integral, zero state and constant-state preservation ({error:.2e})')


if __name__=='__main__':main()
