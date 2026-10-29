"""Portable vanilla CNO for regular-grid Biot fields.

Uses PyTorch bicubic antialiased resize, including oversampled activations.
This is a simplified CNO implementation, not the custom sinc-filtered CNO
from the original paper; no exact representation-equivalence claim is made.
"""
import torch
from torch import nn
import torch.nn.functional as F


def resize_filtered(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    # Float32 interpolation is deliberate for numerical stability under AMP.
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.interpolate(x.float(), size=size, mode="bicubic",
                             align_corners=False, antialias=True).to(x.dtype)


class FilteredActivation(nn.Module):
    def forward(self, x):
        shape = x.shape[-2:]
        fine = resize_filtered(x, (2 * shape[0], 2 * shape[1]))
        return resize_filtered(F.leaky_relu(fine, negative_slope=0.01), shape)


class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                              padding_mode="replicate")
        self.activation = FilteredActivation()

    def forward(self, x):
        return self.activation(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = CNOBlock(channels, channels)
        self.second = nn.Conv2d(channels, channels, 3, padding=1,
                                 padding_mode="replicate")

    def forward(self, x):
        return x + self.second(self.first(x))


class SequentialCNO2d(nn.Module):
    """Input [B,C,H,W] -> next-state correction [B,3,H,W].

    Coordinates are appended internally. ``layers`` is the number of
    downsampling levels, not the number of Fourier layers.
    """
    def __init__(self, input_channels, output_channels=3, width=32,
                 layers=2, padding=8):
        super().__init__()
        if min(input_channels, output_channels, width, layers) < 1 or padding < 0:
            raise ValueError("Channels, width and layers must be positive; padding >= 0")
        if layers > 4:
            raise ValueError("Use 1 to 4 CNO levels to limit feature-map memory")
        self.input_channels, self.output_channels = input_channels, output_channels
        self.width, self.layers, self.padding = width, layers, padding
        channels = [width * 2**i for i in range(layers + 1)]
        self.lift = CNOBlock(input_channels + 2, width)
        self.encoder = nn.ModuleList([ResidualBlock(c) for c in channels[:-1]])
        self.down = nn.ModuleList([CNOBlock(a, b) for a, b in zip(channels[:-1], channels[1:])])
        self.bottleneck = nn.Sequential(ResidualBlock(channels[-1]), ResidualBlock(channels[-1]))
        self.up = nn.ModuleList([CNOBlock(channels[i+1] + channels[i], channels[i])
                                 for i in reversed(range(layers))])
        self.decoder = nn.ModuleList([ResidualBlock(channels[i]) for i in reversed(range(layers))])
        self.project = nn.Conv2d(width, output_channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @staticmethod
    def coordinate_grid(batch, ny, nx, device, dtype):
        y, x = torch.meshgrid(torch.linspace(-1, 1, ny, device=device, dtype=dtype),
                              torch.linspace(-1, 1, nx, device=device, dtype=dtype), indexing="ij")
        return torch.stack((x, y))[None].expand(batch, -1, -1, -1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},H,W], got {tuple(x.shape)}")
        batch, _, ny, nx = x.shape
        if min(ny, nx) < 2**self.layers:
            raise ValueError("Spatial grid too small for requested CNO levels")
        grid = self.coordinate_grid(batch, ny, nx, x.device, x.dtype)
        x = torch.cat((x, grid), dim=1)
        if self.padding:
            x = F.pad(x, (self.padding,) * 4, mode="replicate")
        x = self.lift(x)
        skips = []
        for encoder, down in zip(self.encoder, self.down):
            x = encoder(x)
            skips.append(x)
            # Filtering occurs before subsampling via antialiased interpolation.
            x = resize_filtered(down(x), (x.shape[-2] // 2, x.shape[-1] // 2))
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.up, self.decoder, reversed(skips)):
            x = resize_filtered(x, skip.shape[-2:])
            x = decoder(up(torch.cat((x, skip), dim=1)))
        x = self.project(x)
        if self.padding:
            x = x[..., self.padding:-self.padding, self.padding:-self.padding]
        return x
