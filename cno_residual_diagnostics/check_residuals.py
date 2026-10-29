"""Manufactured-solution, source-sign and aggregation checks on CPU."""
import numpy as np
import torch
from residual_core import physical_terms,ResidualComparison
from biot_physics import spatial_derivatives

torch.set_num_threads(2)
torch.manual_seed(9)
n=16
x=torch.linspace(0,1,n,dtype=torch.float64)
Z,X=torch.meshgrid(x,x,indexing='ij')
A=torch.randn(8,8,dtype=torch.float64)*.01
B=torch.randn(8,8,dtype=torch.float64)*.01
a,b=torch.randn(8,dtype=torch.float64),torch.randn(8,dtype=torch.float64)
slope=-A@a-B@b
def state(t):return (t*slope[:,None,None]+a[:,None,None]*X+b[:,None,None]*Z)[None]
times=torch.tensor([.006],dtype=torch.float64)
kwargs=dict(A=A[None,None].expand(1,2,8,8),B=B[None,None].expand(1,2,8,8),E=torch.zeros(1,2,8,8,dtype=torch.float64),
            ids=torch.zeros(n,n,dtype=torch.long),source=torch.zeros(1,8,n,n,dtype=torch.float64),times=times,
            dt=.002,dx=1/(n-1),dz=1/(n-1),frequency=50.,delay=.04)
ref=physical_terms(state(.006),state(.004),state(.002),**kwargs)
assert ref.sum(1).abs().max()<1e-11
# A constant error in only the predicted next frame produces exactly the
# BDF2 temporal error 3*0.1/(2*dt)=75 physical units/s.
teacher=physical_terms(state(.006)+.1,state(.004),state(.002),**kwargs)
assert torch.allclose((teacher-ref).sum(1),torch.full_like(teacher[:,0],75.),atol=1e-10,rtol=1e-10)
# The same constant offset across every predicted frame has zero bulk
# residual difference here. This deliberately illustrates why field error
# and IC/BC checks must accompany residual consistency.
sequence=physical_terms(state(.006)+.1,state(.004)+.1,state(.002)+.1,**kwargs)
assert (sequence-ref).sum(1).abs().max()<1e-11
valid=torch.ones(n-4,n-4,dtype=torch.bool)
comp=ResidualComparison(8,valid,[1.]*8,.02)
comp.update(np.array([3]),teacher,ref)
summary=comp.summary(tuple(str(c) for c in range(8)))
assert summary['valid_pixel_equation_samples']==(n-4)**2
for row in summary['by_channel'].values():
 assert np.isclose(row['difference_rms_dimensionless'],1.5)
 assert np.isclose(row['cno']['rms_physical'],75.)
 assert row['cno']['relative_balance_rms']<=1+1e-12
# Aggregation over chunks must agree with vectorized evaluation.
twice=ResidualComparison(8,valid,[1.]*8,.02)
twice.update(np.array([3,4]),teacher.expand(2,-1,-1,-1,-1),ref.expand(2,-1,-1,-1,-1))
split=ResidualComparison(8,valid,[1.]*8,.02)
split.update(np.array([3]),teacher,ref);split.update(np.array([4]),teacher,ref)
assert twice.summary(tuple(str(c) for c in range(8)))==split.summary(tuple(str(c) for c in range(8)))
# Ricker at its peak is +1; the signed source contribution is -S.
zeros=torch.zeros(1,8,n,n,dtype=torch.float64)
source=torch.zeros_like(zeros);source[:,5]=2.
source_args=dict(kwargs,source=source,times=torch.tensor([.04],dtype=torch.float64))
r=physical_terms(zeros,zeros,zeros,**source_args).sum(1)
assert torch.equal(r[:,5],torch.full_like(r[:,5],-2.))
assert r[:,:5].abs().sum()==0 and r[:,6:].abs().sum()==0
print('PASS: manufactured Biot solution, BDF2 perturbation, teacher/sequence distinction, scaling, aggregation, source sign')
