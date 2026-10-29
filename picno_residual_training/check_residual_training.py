"""CPU checks of the residual definition, gradients, and RMS aggregation."""
from types import SimpleNamespace
import numpy as np
import torch
from train_residual_picno import ResidualPhysics, summarize


def main():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    n = 24
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    meta = dict(dt=.002, dx=x[1]-x[0], dy=x[1]-x[0], layer_ids=(Y<.5).astype(int),
        projected_source=np.zeros((8, n, n)), y=x, interface_y_m=.5,
        source_frequency_hz=50., source_delay_s=.04)
    args = SimpleNamespace(physics_time_scale=.02, boundary_cells=2, interface_cells=2, source_cells=2)
    stats = dict(output_mean=[0.]*8, output_std=[1.]*8)
    physics = ResidualPhysics(meta, stats, args)
    physics.residual_dtype = torch.float64
    # Exact linear solution of a coupled constant-coefficient first-order PDE.
    A, B = [torch.randn(8, 8, dtype=torch.float64)*.1 for _ in range(2)]
    a, b = [torch.randn(8, dtype=torch.float64) for _ in range(2)]
    slope = -A@a-B@b
    X, Y = torch.tensor(X), torch.tensor(Y)
    def u(t):
        return (t*slope[:, None, None]+a[:, None, None]*X+b[:, None, None]*Y)[None]
    matrices = [m[None, None].expand(1, 2, 8, 8) for m in (A, B, torch.zeros_like(A))]
    t = torch.tensor([.006], dtype=torch.float64)
    r = physics.residual(u(.006), u(.004), u(.002), *matrices, t)
    assert float(physics.bulk_loss(r)) < 1e-23
    predicted = (u(.006)+.1).requires_grad_()
    perturbed = physics.residual(predicted, u(.004), u(.002), *matrices, t)
    assert torch.allclose(perturbed, torch.full_like(perturbed, 1.5), atol=1e-11)
    physics.bulk_loss(perturbed).backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    # MSE/RMS reduce individual residual values, not averages of snapshot RMSs.
    residuals = np.array([[1.]*8, [3.]*8])
    result = summarize((residuals**2).sum(0), 2)
    assert np.isclose(result['mean_rms'], np.sqrt(5))
    assert np.isclose(result['mean_square'], 5)
    mixed = summarize(np.arange(1, 9, dtype=float)**2, 1)
    assert mixed['mean_rms'] == 4.5
    assert np.isclose(mixed['pooled_rms'], np.sqrt(25.5))
    # The source enters with a minus sign.
    physics.source[:, 5] = 2.
    z = torch.zeros_like(u(.006))
    r = physics.residual(z, z, z, *[torch.zeros_like(m) for m in matrices], torch.tensor([.04], dtype=torch.float64))
    assert torch.allclose(r[:, 5], torch.full_like(r[:, 5], -.04))
    print('PASS: coupled PDE, BDF2 scaling, source sign, physics gradient, per-channel and mean RMS aggregation')


if __name__ == '__main__':
    main()
