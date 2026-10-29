"""Dense 2-D Fourier neural operator predicting a next-state correction."""
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Two nonoverlapping y-frequency bands, one-sided real-FFT x frequencies.

    Complex weights are stored as real/imaginary pairs so parameter counts and
    AdamW operations consistently count real scalar degrees of freedom.
    """
    def __init__(self, in_channels, out_channels, modes_y, modes_x):
        super().__init__()
        if min(in_channels, out_channels, modes_y, modes_x) < 1:
            raise ValueError('Channels and mode counts must be positive')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes_y, self.modes_x = modes_y, modes_x
        shape = (in_channels, out_channels, modes_y, modes_x, 2)
        scale = 1/(in_channels*out_channels)
        self.weight_positive = nn.Parameter(scale*torch.randn(shape))
        self.weight_negative = nn.Parameter(scale*torch.randn(shape))

    def forward(self, x):
        batch, _, ny, nx = x.shape
        if 2*self.modes_y > ny or self.modes_x > nx//2+1:
            raise ValueError(f'Modes ({self.modes_y},{self.modes_x}) exceed padded grid ({ny},{nx}); '
                             'require 2*modes_y <= ny and modes_x <= nx//2+1')
        # cuFFT half precision is restricted on non-power-of-two grids.
        # Always FFT and perform spectral multiplication in float32/complex64.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='ortho')
            out = torch.zeros(batch, self.out_channels, ny, nx//2+1,
                              device=x.device, dtype=torch.complex64)
            wp = torch.view_as_complex(self.weight_positive.float().contiguous())
            wn = torch.view_as_complex(self.weight_negative.float().contiguous())
            my, mx = self.modes_y, self.modes_x
            out[:, :, :my, :mx] = torch.einsum('biyx,ioyx->boyx', spectrum[:, :, :my, :mx], wp)
            out[:, :, -my:, :mx] = torch.einsum('biyx,ioyx->boyx', spectrum[:, :, -my:, :mx], wn)
            return torch.fft.irfft2(out, s=(ny, nx), norm='ortho')


class SequentialFNO2d(nn.Module):
    """Same external mapping as CNO; `layers` counts Fourier blocks here."""
    def __init__(self, input_channels, output_channels=8, width=32, layers=2,
                 padding=8, modes_y=22, modes_x=22):
        super().__init__()
        if min(input_channels, output_channels, width, layers) < 1 or padding < 0:
            raise ValueError('Positive channels/width/layers and nonnegative padding required')
        self.input_channels, self.padding = input_channels, padding
        self.lift = nn.Conv2d(input_channels+2, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes_y, modes_x) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project_hidden = nn.Conv2d(width, width, 1)
        self.project = nn.Conv2d(width, output_channels, 1)
        # Same baseline at initialization as the CNO zero correction head.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(f'Expected [B,{self.input_channels},H,W]')
        b, _, ny, nx = inputs.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, ny, device=inputs.device, dtype=inputs.dtype),
                              torch.linspace(-1, 1, nx, device=inputs.device, dtype=inputs.dtype), indexing='ij')
        coords = torch.stack((x, y))[None].expand(b, -1, -1, -1)
        value = torch.cat((inputs, coords), dim=1)
        if self.padding:
            value = F.pad(value, (self.padding,)*4, mode='replicate')
        value = self.lift(value)
        for spectral, local in zip(self.spectral, self.local):
            value = F.gelu(spectral(value)+local(value))
        value = self.project(F.gelu(self.project_hidden(value)))
        if self.padding:
            value = value[..., self.padding:-self.padding, self.padding:-self.padding]
        return value
