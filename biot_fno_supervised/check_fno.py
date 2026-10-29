"""CPU verification of spectral multiplication, truncation, shapes and gradients."""
import torch
from fno_model import SpectralConv2d, SequentialFNO2d


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    for shape in ((16, 18), (17, 19)):
        ny, nx = shape
        layer = SpectralConv2d(2, 3, 4, 5)
        x = torch.randn(2, 2, ny, nx, requires_grad=True)
        result = layer(x)
        # Independent real/imaginary reference, summing each input channel.
        transformed = torch.fft.rfft2(x, norm='ortho')
        ref = torch.zeros(2, 3, ny, nx//2+1, dtype=torch.complex64)
        for region, weights in ((slice(0, 4), layer.weight_positive), (slice(-4, None), layer.weight_negative)):
            for i in range(2):
                w = weights[i, ..., 0]+1j*weights[i, ..., 1]
                ref[:, :, region, :5] += transformed[:, i:i+1, region, :5]*w[None]
        expected = torch.fft.irfft2(ref, s=shape, norm='ortho')
        assert torch.allclose(result, expected, atol=1e-6, rtol=1e-5)
        result.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        for weight in (layer.weight_positive, layer.weight_negative):
            assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    # Identity on retained low modes, zero on discarded pure Fourier modes.
    layer = SpectralConv2d(1, 1, 4, 5)
    with torch.no_grad():
        layer.weight_positive.zero_(); layer.weight_negative.zero_()
        layer.weight_positive[..., 0] = 1
        layer.weight_negative[..., 0] = 1
    y, x = torch.meshgrid(torch.arange(24), torch.arange(26), indexing='ij')
    low = torch.cos(2*torch.pi*(2*y/24+3*x/26))[None, None]
    high = torch.cos(2*torch.pi*(7*y/24+8*x/26))[None, None]
    assert torch.allclose(layer(low), low, atol=5e-6)
    assert layer(high).abs().max() < 5e-6
    try:
        SpectralConv2d(1, 1, 9, 2)(torch.zeros(1, 1, 16, 18))
    except ValueError:
        pass
    else:
        raise AssertionError('Overlapping mode bands must be rejected')
    model = SequentialFNO2d(44, width=4, layers=2, padding=3, modes_y=4, modes_x=5)
    x = torch.randn(2, 44, 17, 19)
    correction = model(x)
    assert correction.shape == (2, 8, 17, 19) and torch.equal(correction, torch.zeros_like(correction))
    # The first step trains the zero output head; subsequent steps reach spectral weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn_like(correction)
    for _ in range(2):
        optimizer.zero_grad()
        (model(x)-target).square().mean().backward()
        optimizer.step()
    assert model.spectral[0].weight_positive.grad.abs().sum() > 0
    counts = {m: sum(p.numel() for p in SequentialFNO2d(44, modes_y=m, modes_x=m).parameters()) for m in (16, 22)}
    print('PASS: complex spectral multiplication, mode truncation, odd/even grids, nonoverlap, shape, zero-head baseline, spectral gradients')
    print('Real parameter counts (width=32, layers=2):', counts)


if __name__ == '__main__':
    main()
