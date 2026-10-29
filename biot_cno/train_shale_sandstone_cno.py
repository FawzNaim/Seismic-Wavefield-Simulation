"""One-step-only vanilla CNO for the existing Biot ADER-DG HDF5 dataset.

By default all predicted frames use four TRUE ADER-DG history frames. There is no
prediction feedback, recursive training, curriculum, or recursive evaluation.
The CNO architecture and normalized residual prediction are unchanged:
q_hat[n] = 2*q_true[n-1] - q_true[n-2] + CNO(inputs).
State q = [p, vx, vy], where vx/vy are solid velocities, not displacement.
Inputs: material maps, shared source map, target time, four true states, coordinates.
History order: q[n-4], q[n-3], q[n-2], q[n-1] -> q_hat[n].
All 12 history channels (4 times x 3 fields) enter CNO.
The residual baseline uses only the last two frames to preserve the earlier
comparison design. --history-frames 2 enables the old two-frame configuration.
Train-only normalization; model selection uses validation one-step relative L2.
Loss: supervised wavefield matching + gradient matching + RMS matching.
This version is data-trained and contains no acoustic/Biot physics loss.

Dependencies: Python >=3.10; torch >=2.2, numpy, h5py, matplotlib.
Use your workstation's CUDA-enabled PyTorch environment.

Example:
python -u train_shale_sandstone_cno.py --data DATA.h5 \
  --output-dir cno_four_frame_results --epochs 150 --no-early-stopping \
  --batch-size 2 --width 32 --layers 2 --windows-per-realization 16

--windows-per-realization counts independent ONE-STEP pairs sampled per
training realization each epoch, not consecutive prediction steps. Set 0 to
use all available pairs. Validation/test always use all available true pairs.
--overfit-one fits/monitors training realization 0, a diagnostic only.
--evaluate-paper is opt-in and never affects model selection.

Four-frame training requires a NEW checkpoint because the lifting layer has
six additional input channels. Existing two-frame checkpoints can be evaluated
only with --history-frames 2. No changes to HDF5 data are needed.
A model trained with one-step-only selection may choose a different best epoch.
Checkpoints support evaluation, not exact optimizer/scheduler resumption.

Required HDF5: coords/x_m, y_m, times_s; static/source_map;
metadata/input_names, output_names; train/validation/test each with
inputs [N,C,H,W], wavefields [N,T,3,H,W], completed [N] all equal 1.
Saved times must be uniform. Native triangular DOFs must be sampled to a
regular grid first. Predictions are conditional next-frame estimates, not
independent simulations from initial conditions to arbitrary target times.
The first history_frames frames are copied seeds and excluded from metrics.

This portable CNO uses bicubic antialiased resize and oversampled activation;
it is not an exact replica of the original custom sinc-filtered architecture.
Source: https://github.com/camlab-ethz/ConvolutionalNeuralOperator
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_strings(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def verify_dataset(path: Path, history_frames=4) -> dict:
    """Fail early if the HDF5 file is incomplete, truncated, or incompatible."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise OSError(
            f"Cannot open {path}. The HDF5 file may be truncated or may still "
            "be open for writing by the dataset generator."
        ) from exc

    with handle as f:
        required = ["coords", "static", "metadata", "train", "validation", "test"]
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"Missing HDF5 groups: {missing}")

        input_names = decode_strings(f["metadata/input_names"][:])
        output_names = decode_strings(f["metadata/output_names"][:])
        if output_names != ["p", "vx", "vy"]:
            raise ValueError(
                f"Expected output channels ['p', 'vx', 'vy']; found {output_names}"
            )

        times = f["coords/times_s"][:]
        if len(times) < history_frames + 1:
            raise ValueError(
                f"Need at least {history_frames + 1} saved frames for {history_frames}-frame inputs"
            )
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("coords/times_s must be strictly increasing")
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-4, atol=1e-8):
            raise ValueError("One-step CNO extrapolation requires uniformly spaced saved times")
        x = f["coords/x_m"][:]
        y = f["coords/y_m"][:]
        split_shapes = {}
        for split in ("train", "validation", "test"):
            completed = f[f"{split}/completed"][:]
            if len(completed) == 0:
                raise ValueError(f"The {split} split is empty")
            if not np.all(completed == 1):
                unfinished = np.where(completed != 1)[0].tolist()
                raise ValueError(
                    f"The {split} split contains unfinished samples: {unfinished}"
                )
            inputs = f[f"{split}/inputs"]
            outputs = f[f"{split}/wavefields"]
            expected_inputs = (len(completed), len(input_names), len(y), len(x))
            expected_outputs = (
                len(completed), len(times), len(output_names), len(y), len(x)
            )
            if inputs.shape != expected_inputs:
                raise ValueError(
                    f"Unexpected {split}/inputs shape {inputs.shape}; "
                    f"expected {expected_inputs}"
                )
            if outputs.shape != expected_outputs:
                raise ValueError(
                    f"Unexpected {split}/wavefields shape {outputs.shape}; "
                    f"expected {expected_outputs}"
                )
            split_shapes[split] = {
                "realizations": len(completed),
                "pairs": len(completed) * (len(times) - history_frames),
            }

        information = {
            "input_names": input_names,
            "output_names": output_names,
            "times": times,
            "x": x,
            "y": y,
            "splits": split_shapes,
            "has_paper_reference": "paper_reference" in f,
        }

    print("Dataset verified")
    print(f"  grid                 : {len(y)} x {len(x)}")
    print(f"  saved times          : {len(times)} ({times[0]:.3f}--{times[-1]:.3f} s)")
    for split, values in split_shapes.items():
        print(
            f"  {split:10s}           : {values['realizations']} realizations, "
            f"{values['pairs']} independent history-to-next-frame examples"
        )
    print(f"  paper reference      : {information['has_paper_reference']}")
    return information


def safe_standard_deviation(total, total_sq, count):
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(variance)
    # Constant channels are left centered but not amplified.
    std = np.where(std < 1.0e-12 * np.maximum(np.abs(mean), 1.0), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_training_statistics(path: Path) -> dict:
    """Calculate all means/stds from train only, without loading the file at once."""
    print("Computing train-only normalization statistics...")
    with h5py.File(path, "r") as f:
        inputs = f["train/inputs"]
        outputs = f["train/wavefields"]

        input_sum = np.zeros(inputs.shape[1], dtype=np.float64)
        input_sq = np.zeros_like(input_sum)
        input_count = 0
        for realization in range(inputs.shape[0]):
            array = np.asarray(inputs[realization], dtype=np.float64)
            input_sum += np.sum(array, axis=(1, 2))
            input_sq += np.sum(array**2, axis=(1, 2))
            input_count += array.shape[1] * array.shape[2]

        output_sum = np.zeros(outputs.shape[2], dtype=np.float64)
        output_sq = np.zeros_like(output_sum)
        output_count = 0
        for realization in range(outputs.shape[0]):
            for time_index in range(outputs.shape[1]):
                array = np.asarray(outputs[realization, time_index], dtype=np.float64)
                if not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"NaN/Inf in train realization {realization}, "
                        f"time index {time_index}"
                    )
                output_sum += np.sum(array, axis=(1, 2))
                output_sq += np.sum(array**2, axis=(1, 2))
                output_count += array.shape[1] * array.shape[2]

    input_mean, input_std = safe_standard_deviation(
        input_sum, input_sq, input_count
    )
    output_mean, output_std = safe_standard_deviation(
        output_sum, output_sq, output_count
    )
    print("  output means:", output_mean)
    print("  output stds :", output_std)
    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
    }


class SequentialH5Dataset(Dataset):
    """One item = H true history states -> one target state; oldest first."""

    def __init__(self, path, split, statistics, realization_indices=None, history_frames=4):
        self.history_frames = int(history_frames)
        if self.history_frames < 2:
            raise ValueError("At least two history frames are required")
        self.path = str(path)
        self.split = split
        self.input_mean = statistics["input_mean"].astype(np.float32)[:, None, None]
        self.input_std = statistics["input_std"].astype(np.float32)[:, None, None]
        self.output_mean = statistics["output_mean"].astype(np.float32)[:, None, None]
        self.output_std = statistics["output_std"].astype(np.float32)[:, None, None]
        self._handle = None

        with h5py.File(self.path, "r") as f:
            available_realizations = f[f"{split}/inputs"].shape[0]
            self.times = f["coords/times_s"][:].astype(np.float32)
            self.final_time = float(max(self.times[-1], 1.0e-12))
            self.source_map = f["static/source_map"][:].astype(np.float32)
            self.ny, self.nx = self.source_map.shape
        if realization_indices is None:
            self.realizations = list(range(available_realizations))
        else:
            self.realizations = [int(value) for value in realization_indices]
            if any(
                value < 0 or value >= available_realizations
                for value in self.realizations
            ):
                raise IndexError(
                    f"Invalid realization selection for split {split}: "
                    f"{self.realizations}"
                )
        self.n_realizations = len(self.realizations)
        if len(self.times) < self.history_frames + 1:
            raise ValueError(f"Need at least {self.history_frames + 1} saved frames")
        self.index = [
            (realization, target_index)
            for realization in self.realizations
            for target_index in range(self.history_frames, len(self.times))
        ]

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        realization, target_index = self.index[item]
        f = self._file()
        properties = np.asarray(
            f[f"{self.split}/inputs"][realization], dtype=np.float32
        )
        history = np.asarray(
            f[f"{self.split}/wavefields"][realization, target_index-self.history_frames:target_index],
            dtype=np.float32,
        )
        target = np.asarray(
            f[f"{self.split}/wavefields"][realization, target_index],
            dtype=np.float32,
        )
        properties = (properties - self.input_mean) / self.input_std
        history = (history - self.output_mean[None]) / self.output_std[None]
        history = history.reshape(-1, self.ny, self.nx)
        target = (target - self.output_mean) / self.output_std

        time_value = self.times[target_index] / self.final_time
        time_map = np.full((self.ny, self.nx), time_value, dtype=np.float32)
        model_input = np.concatenate(
            (
                properties,
                self.source_map[None],
                time_map[None],
                history,
            ),
            axis=0,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(model_input)),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.tensor(realization, dtype=torch.long),
            torch.tensor(target_index, dtype=torch.long),
        )


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


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def wavefield_balanced_loss(predictions, targets, active_fraction=0.05):
    """Stable loss for sparse wavefields without small-energy division.

    The strongest target pixels define a wavefront-focused term.  A global
    term prevents false energy elsewhere.  Smooth L1 is quadratic near zero
    and linear for large errors, preventing rare amplitudes from causing the
    explosive gradients seen with relative-energy denominators.
    """
    pixel_loss = F.smooth_l1_loss(
        predictions, targets, reduction="none", beta=0.10
    )
    batch, channels, ny, nx = targets.shape
    spatial_size = ny * nx
    active_count = max(1, int(round(active_fraction * spatial_size)))
    target_flat = torch.abs(targets).reshape(batch, channels, spatial_size)
    loss_flat = pixel_loss.reshape(batch, channels, spatial_size)
    active_indices = torch.topk(
        target_flat, k=active_count, dim=-1, largest=True, sorted=False
    ).indices
    active_loss = torch.gather(loss_flat, dim=-1, index=active_indices).mean()
    global_loss = pixel_loss.mean()
    return 0.70 * active_loss + 0.30 * global_loss


def spatial_gradient_loss(predictions, targets):
    """Penalize blurred or misplaced wavefront edges in normalized units."""
    pred_dx = predictions[..., :, 1:] - predictions[..., :, :-1]
    true_dx = targets[..., :, 1:] - targets[..., :, :-1]
    pred_dy = predictions[..., 1:, :] - predictions[..., :-1, :]
    true_dy = targets[..., 1:, :] - targets[..., :-1, :]
    return 0.5 * (
        F.smooth_l1_loss(pred_dx, true_dx, beta=0.10)
        + F.smooth_l1_loss(pred_dy, true_dy, beta=0.10)
    )


def wave_energy_loss(predictions, targets, minimum_target_rms=0.02):
    """Keep recursive predictions from satisfying the PDE by fading to zero."""
    predicted_rms = torch.sqrt(torch.mean(predictions.float() ** 2, dim=(2, 3)) + 1.0e-8)
    target_rms = torch.sqrt(torch.mean(targets.float() ** 2, dim=(2, 3)) + 1.0e-8)
    valid = (target_rms >= minimum_target_rms).float()
    loss = F.smooth_l1_loss(
        predicted_rms, target_rms, beta=0.10, reduction="none"
    )
    return torch.sum(loss * valid) / torch.clamp(torch.sum(valid), min=1.0)


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_one_step_loader(
    model, loader, statistics, device, times, output_names, amp
):
    """Teacher-forced error: every history state comes from ADER-DG."""
    model.eval()
    output_mean = torch.as_tensor(
        statistics["output_mean"], device=device, dtype=torch.float32
    )[None, :, None, None]
    output_std = torch.as_tensor(
        statistics["output_std"], device=device, dtype=torch.float32
    )[None, :, None, None]
    nt = len(times)
    nc = len(output_names)
    error_sq = np.zeros((nt, nc), dtype=np.float64)
    truth_sq = np.zeros((nt, nc), dtype=np.float64)

    for inputs, targets_normalized, _, time_indices in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets_normalized = targets_normalized.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(amp and device.type == "cuda"),
        ):
            correction = model(inputs)
        channels = targets_normalized.shape[1]
        previous = inputs[:, -2 * channels : -channels].float()
        current = inputs[:, -channels:].float()
        predictions_normalized = 2.0 * current - previous + correction.float()
        predictions = predictions_normalized.float() * output_std + output_mean
        targets = targets_normalized.float() * output_std + output_mean
        difference = predictions - targets
        batch_error = torch.sum(difference**2, dim=(2, 3)).cpu().numpy()
        batch_truth = torch.sum(targets**2, dim=(2, 3)).cpu().numpy()
        for row, time_index in enumerate(time_indices.numpy()):
            error_sq[time_index] += batch_error[row]
            truth_sq[time_index] += batch_truth[row]

    total_error = np.sum(error_sq, axis=0)
    total_truth = np.sum(truth_sq, axis=0)
    relative_by_channel = np.sqrt(total_error / np.maximum(total_truth, 1.0e-30))
    relative_by_time_channel = np.sqrt(
        error_sq / np.maximum(truth_sq, 1.0e-30)
    )
    # Average the dimensionless channel-wise squared relative errors.  Summing
    # raw Pa and m/s energies together would let pressure dominate by units.
    relative_overall = math.sqrt(
        float(np.mean(total_error / np.maximum(total_truth, 1.0e-30)))
    )
    return {
        "relative_overall": relative_overall,
        "relative_by_channel": relative_by_channel,
        "relative_by_time_channel": relative_by_time_channel,
        "first_predicted_index": model.history_frames,
        "history_frames": model.history_frames,
        "evaluation": "teacher_forced_one_step",
    }


def normalized_sequential_input(properties, source_map, time_value, history_normalized, statistics):
    mean = statistics["input_mean"][:, None, None]
    std = statistics["input_std"][:, None, None]
    properties = (properties.astype(np.float32) - mean) / std
    time_map = np.full(source_map.shape, time_value, dtype=np.float32)
    history_flat = history_normalized.astype(np.float32).reshape(-1, *source_map.shape)
    return np.concatenate((properties, source_map[None].astype(np.float32), time_map[None], history_flat), axis=0)


@torch.no_grad()
def one_step_realization(
    model, data_path, split, realization, statistics, device, amp
):
    """Predict each eligible frame independently from H true history states."""
    with h5py.File(data_path, "r") as f:
        times = f["coords/times_s"][:].astype(np.float32)
        source = f["static/source_map"][:].astype(np.float32)
        inputs_ds = f[f"{split}/inputs"]
        fields_ds = f[f"{split}/wavefields"]
        if realization is None and inputs_ds.ndim == 3 and fields_ds.ndim == 4:
            properties = inputs_ds[:].astype(np.float32)
            reference = fields_ds[:].astype(np.float32)
        else:
            position = 0 if realization is None else realization
            properties = inputs_ds[position].astype(np.float32)
            reference = fields_ds[position].astype(np.float32)
    output_mean = statistics["output_mean"][:, None, None].astype(np.float32)
    output_std = statistics["output_std"][:, None, None].astype(np.float32)
    normalized = (reference - output_mean[None]) / output_std[None]
    prediction = reference.copy()
    final_time = max(float(times[-1]), 1.0e-12)
    model.eval()
    if len(times) <= model.history_frames:
        raise ValueError("Not enough saved frames for checkpoint history length")
    for target_index in range(model.history_frames, len(times)):
        history = normalized[target_index-model.history_frames:target_index]
        previous = history[-2]
        current = history[-1]
        model_input = normalized_sequential_input(
            properties,
            source,
            float(times[target_index] / final_time),
            history,
            statistics,
        )
        tensor = torch.from_numpy(model_input[None]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(amp and device.type == "cuda"),
        ):
            correction = model(tensor).float().cpu().numpy()[0]
        next_normalized = 2.0 * current - previous + correction
        prediction[target_index] = next_normalized * output_std + output_mean
    return times, reference, prediction


def trajectory_metrics(reference, prediction, first_predicted_index=4):
    """Relative physical-unit metrics for one or more full trajectories."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.ndim == 4:
        reference = reference[None]
        prediction = prediction[None]
    if reference.shape != prediction.shape or reference.ndim != 5:
        raise ValueError(
            "Expected matching [realization,time,channel,y,x] trajectories"
        )
    _, nt, nc, _, _ = reference.shape
    error_sq = np.zeros((nt, nc), dtype=np.float64)
    truth_sq = np.zeros((nt, nc), dtype=np.float64)
    difference = (
        prediction[:, first_predicted_index:]
        - reference[:, first_predicted_index:]
    )
    error_sq[first_predicted_index:] = np.sum(
        difference**2, axis=(0, 3, 4)
    )
    truth_sq[first_predicted_index:] = np.sum(
        reference[:, first_predicted_index:] ** 2, axis=(0, 3, 4)
    )
    total_error = np.sum(error_sq[first_predicted_index:], axis=0)
    total_truth = np.sum(truth_sq[first_predicted_index:], axis=0)
    relative_by_channel = np.sqrt(
        total_error / np.maximum(total_truth, 1.0e-30)
    )
    relative_by_time_channel = np.full((nt, nc), np.nan, dtype=np.float64)
    relative_by_time_channel[first_predicted_index:] = np.sqrt(
        error_sq[first_predicted_index:]
        / np.maximum(truth_sq[first_predicted_index:], 1.0e-30)
    )
    relative_overall = math.sqrt(
        float(np.mean(total_error / np.maximum(total_truth, 1.0e-30)))
    )
    return {
        "relative_overall": relative_overall,
        "relative_by_channel": relative_by_channel,
        "relative_by_time_channel": relative_by_time_channel,
        "first_predicted_index": first_predicted_index,
        "evaluation": "teacher_forced_one_step",
    }


def plot_error_vs_time(metrics, times, output_names, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    errors = metrics["relative_by_time_channel"]
    first = int(metrics.get("first_predicted_index", 2))
    for channel, name in enumerate(output_names):
        values = np.maximum(errors[first:, channel], 1.0e-8)
        ax.semilogy(times[first:], values, label=name)
    ax.set(
        xlabel="Time (s)",
        ylabel="Relative $L_2$ error",
        title=f"{prefix.replace('_', ' ').title()} error through time",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    path = output_dir / f"{prefix}_error_vs_time.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_field_comparison(
    reference, prediction, times, output_names, output_dir, prefix, requested_time=0.25, first_predicted_index=4
):
    time_index = int(np.argmin(np.abs(times - requested_time)))
    if time_index < first_predicted_index:
        raise ValueError(f"Requested snapshot is a seed frame; choose --plot-time >= {times[first_predicted_index]:.6g} s")
    fig, axes = plt.subplots(
        len(output_names), 3, figsize=(14, 4.1 * len(output_names)),
        constrained_layout=True
    )
    for channel, name in enumerate(output_names):
        truth = reference[time_index, channel]
        estimate = prediction[time_index, channel]
        difference = estimate - truth
        bound = max(float(np.percentile(np.abs(truth), 99.5)), 1.0e-30)
        error_bound = max(float(np.percentile(np.abs(difference), 99.5)), 1.0e-30)
        a = axes[channel, 0].imshow(
            truth, origin="lower", cmap="seismic", vmin=-bound, vmax=bound,
            aspect="auto"
        )
        b = axes[channel, 1].imshow(
            estimate, origin="lower", cmap="seismic", vmin=-bound, vmax=bound,
            aspect="auto"
        )
        c = axes[channel, 2].imshow(
            difference, origin="lower", cmap="seismic",
            vmin=-error_bound, vmax=error_bound, aspect="auto"
        )
        axes[channel, 0].set_ylabel(name)
        for axis, title in zip(
            axes[channel], ("ADER-DG", "CNO", "CNO - ADER-DG")
        ):
            axis.set_title(title)
            axis.set_xlabel("x-grid index")
        fig.colorbar(a, ax=axes[channel, 0], shrink=0.78)
        fig.colorbar(b, ax=axes[channel, 1], shrink=0.78)
        fig.colorbar(c, ax=axes[channel, 2], shrink=0.78)
    fig.suptitle(
        f"{prefix.replace('_', ' ').title()} at t={times[time_index]:.2f} s",
        fontsize=15,
    )
    path = output_dir / f"{prefix}_comparison_t{times[time_index]:.2f}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics, output_names, path):
    serializable = {
        "relative_overall": float(metrics["relative_overall"]),
        "relative_by_channel": {
            name: float(metrics["relative_by_channel"][index])
            for index, name in enumerate(output_names)
        },
    }
    for key in ("evaluation", "first_predicted_index", "realizations_evaluated", "history_frames"):
        if key in metrics:
            value = metrics[key]
            serializable[key] = int(value) if isinstance(
                value, (np.integer, int)
            ) else value
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))


def predict_one_step(model, inputs, channels, amp):
    with torch.autocast(device_type=inputs.device.type, dtype=torch.float16,
                        enabled=amp and inputs.device.type == 'cuda'):
        correction = model(inputs)
    previous = inputs[:, -2*channels:-channels].float()
    current = inputs[:, -channels:].float()
    return 2.0 * current - previous + correction.float()


def run_one_step_epoch(model, loader, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(4, dtype=np.float64)
    count = 0
    for inputs, target, _, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if not torch.isfinite(inputs).all() or not torch.isfinite(target).all():
            raise ValueError('Nonfinite input or target in dataset')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = predict_one_step(model, inputs, target.shape[1], args.amp)
            field = wavefield_balanced_loss(prediction, target.float())
            gradient = spatial_gradient_loss(prediction, target.float())
            energy = wave_energy_loss(prediction, target.float())
            loss = field + args.gradient_loss_weight * gradient + args.energy_loss_weight * energy
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite one-step loss; check labels or reduce learning rate')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        totals += np.array([v.detach().item() for v in (loss, field, gradient, energy)]) * inputs.shape[0]
        count += inputs.shape[0]
    if count == 0:
        raise ValueError('No one-step pairs available')
    return dict(zip(('loss', 'field', 'gradient', 'energy'), totals / count))


def make_loader(dataset, args, device, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=device.type == 'cuda')


def sampled_training_pairs(dataset, limit, epoch, seed):
    if limit == 0 or limit >= len(dataset.times)-dataset.history_frames:
        return dataset
    rng = np.random.default_rng(seed + epoch * 104729)
    positions = []
    pairs_per_realization = len(dataset.times)-dataset.history_frames
    for position, _ in enumerate(dataset.realizations):
        candidates = np.arange(position*pairs_per_realization, (position+1)*pairs_per_realization)
        positions.extend(rng.choice(candidates, min(limit, len(candidates)), replace=False).tolist())
    return Subset(dataset, positions)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('architecture') not in ('vanilla_residual_cno_v1', 'vanilla_one_step_cno_v2', 'vanilla_one_step_cno_history_v3'):
        raise ValueError('Requires a compatible vanilla CNO checkpoint, not an FNO checkpoint')
    model = SequentialCNO2d(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    statistics = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint['statistics'].items()}
    static_channels = len(statistics['input_mean']) + 2
    extra = model.input_channels - static_channels
    if extra < 2 * model.output_channels or extra % model.output_channels:
        raise ValueError("Checkpoint history channels are invalid")
    inferred_history = extra // model.output_channels
    model.history_frames = int(checkpoint.get('history_frames', inferred_history))
    if model.history_frames != inferred_history:
        raise ValueError("Checkpoint history metadata conflicts with lifting-layer size")
    return model, statistics, checkpoint


def plot_training_history(history, output_dir):
    epochs = np.arange(1, len(history['train_loss'])+1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].semilogy(epochs, history['train_loss'], label='Train one-step loss')
    axes[0].semilogy(epochs, history['validation_loss'], label='Validation one-step loss')
    axes[0].set(xlabel='Epoch', ylabel='Supervised loss', title='CNO one-step training')
    axes[1].semilogy(epochs, history['validation_one_step_relative'], label='Validation one-step relative L2')
    axes[1].set(xlabel='Epoch', ylabel='Relative L2 error', title='Checkpoint selection: one-step error')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(output_dir/'01_training_history.png', dpi=180)
    plt.close(fig)


def train_model(args, model, statistics, train_dataset, validation_dataset, information, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=max(2,args.patience//3))
    scaler = make_grad_scaler(args.amp)
    validation_loader = make_loader(validation_dataset, args, device)
    history = {'train_loss': [], 'validation_loss': [], 'validation_one_step_relative': [], 'learning_rate': []}
    best = float('inf'); stale = 0
    best_path, final_path = args.output_dir/'best_cno.pt', args.output_dir/'final_cno.pt'
    def payload(epoch, relative):
        return {'architecture': 'vanilla_one_step_cno_history_v3', 'history_frames': args.history_frames, 'model_state': model.state_dict(),
                'model_config': {'input_channels': model.input_channels, 'output_channels': model.output_channels,
                                 'width': model.width, 'layers': model.layers, 'padding': model.padding},
                'statistics': statistics, 'epoch': epoch, 'validation_one_step_error': relative,
                'selection_metric': 'validation_teacher_forced_one_step_relative_L2', 'history': history,
                'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    print(f'Training one-step CNO with {args.history_frames} true ADER-DG history frames')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; device: {device}; AMP: {args.amp}')
    for epoch in range(1,args.epochs+1):
        started = time.perf_counter()
        training_pairs = sampled_training_pairs(train_dataset,args.windows_per_realization,epoch,args.seed)
        train_result = run_one_step_epoch(model,make_loader(training_pairs,args,device,True),device,args,optimizer,scaler)
        validation_result = run_one_step_epoch(model,validation_loader,device,args)
        metrics = evaluate_one_step_loader(model, validation_loader, statistics, device,
                                          information['times'], information['output_names'], args.amp)
        relative = float(metrics['relative_overall'])
        if not math.isfinite(relative):
            raise RuntimeError('Nonfinite validation one-step error')
        scheduler.step(relative)
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(float(train_result['loss']))
        history['validation_loss'].append(float(validation_result['loss']))
        history['validation_one_step_relative'].append(relative)
        history['learning_rate'].append(lr)
        improved = relative < best-args.minimum_delta
        if improved:
            best,stale = relative,0
            atomic_torch_save(payload(epoch,relative),best_path)
        else:
            stale += 1
        print(f"epoch {epoch:03d} | train={train_result['loss']:.5e} | val={validation_result['loss']:.5e} | val-one-step={relative:.5e} | pairs={len(training_pairs)} | lr={lr:.2e} | {time.perf_counter()-started:.1f}s" + (' *' if improved else ''),flush=True)
        atomic_torch_save(payload(epoch,relative),final_path)
        if not args.no_early_stopping and stale>=args.patience:
            print(f'Early stopping at epoch {epoch}')
            break
    selected = final_path if args.evaluate_final else best_path
    print(f'Selected checkpoint: {selected}')
    return selected,history


def export_one_step(model, args, statistics, information, device, split, realization=0, prefix='test'):
    if split == 'paper_reference':
        times,reference,prediction = one_step_realization(model,args.data,split,None,statistics,device,args.amp)
        metrics = trajectory_metrics(reference,prediction,first_predicted_index=model.history_frames)
        metrics["history_frames"] = model.history_frames
    else:
        dataset = SequentialH5Dataset(args.data,split,statistics,
                                      realization_indices=[0] if args.overfit_one else None, history_frames=model.history_frames)
        metrics = evaluate_one_step_loader(model,make_loader(dataset,args,device),statistics,device,
                                           information['times'],information['output_names'],args.amp)
        times,reference,prediction = one_step_realization(model,args.data,split,realization,statistics,device,args.amp)
    save_metrics(metrics,information['output_names'],args.output_dir/f'{prefix}_one_step_metrics.json')
    plot_error_vs_time(metrics,times,information['output_names'],args.output_dir,f'{prefix}_one_step')
    sample_prefix = f'{prefix}_sample_0_one_step'
    plot_field_comparison(reference,prediction,times,information['output_names'],args.output_dir,sample_prefix,args.plot_time,first_predicted_index=model.history_frames)
    np.savez_compressed(args.output_dir/f'{sample_prefix}_predictions.npz',times_s=times,reference=reference,
                        prediction=prediction,output_names=np.asarray(information['output_names']),
                        first_predicted_index=model.history_frames,history_frames=model.history_frames,evaluation='teacher_forced_one_step')


def parse_arguments():
    p=argparse.ArgumentParser(description='One-step-only CNO: true ADER-DG histories for all predictions')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('cno_four_frame_results'))
    p.add_argument('--history-frames', type=int, choices=[2,4], default=4, help='True history frames per input (default 4); output remains one frame')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--padding',type=int,default=8)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--minimum-delta',type=float,default=1e-6)
    p.add_argument('--gradient-loss-weight',type=float,default=.1)
    p.add_argument('--energy-loss-weight',type=float,default=.1)
    p.add_argument('--windows-per-realization',type=int,default=16,help='Independent one-step training pairs per realization; 0 uses all')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--no-early-stopping',action='store_true')
    p.add_argument('--no-amp',dest='amp',action='store_false')
    p.set_defaults(amp=True)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--evaluate-final',action='store_true')
    p.add_argument('--evaluate-paper',action='store_true')
    p.add_argument('--overfit-one',action='store_true')
    p.add_argument('--plot-time',type=float,default=.25)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.patience)<1 or args.learning_rate<=0:
        p.error('epochs, batch-size, patience and learning-rate must be positive')
    if min(args.windows_per_realization,args.workers,args.weight_decay,args.gradient_loss_weight,args.energy_loss_weight,args.minimum_delta)<0:
        p.error('Sampling counts, workers and loss/optimizer settings cannot be negative')
    return args


def main():
    args=parse_arguments();args.output_dir.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed)
    information=verify_dataset(args.data,args.history_frames)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    args.amp=args.amp and device.type=='cuda'
    channels=len(information['input_names'])+2+args.history_frames*len(information['output_names'])
    if args.evaluate_only:
        if args.checkpoint is None:
            raise ValueError('--evaluate-only requires --checkpoint')
        model,statistics,checkpoint=load_checkpoint(args.checkpoint,device)
        if model.input_channels!=channels:
            raise ValueError(f'Checkpoint uses {model.history_frames} history frames but --history-frames is {args.history_frames}. Train a new four-frame model, or pass --history-frames 2 to evaluate old weights.')
        history=checkpoint.get('history')
        if history and 'train_loss' not in history:
            history=None  # Legacy training history used a different selection criterion.
    else:
        statistics=compute_training_statistics(args.data)
        model=SequentialCNO2d(channels,len(information['output_names']),args.width,args.layers,args.padding).to(device)
        model.history_frames=args.history_frames
        train=SequentialH5Dataset(args.data,'train',statistics,[0] if args.overfit_one else None,history_frames=args.history_frames)
        validation=SequentialH5Dataset(args.data,'train' if args.overfit_one else 'validation',statistics,[0] if args.overfit_one else None,history_frames=args.history_frames)
        path,history=train_model(args,model,statistics,train,validation,information,device)
        model,statistics,_=load_checkpoint(path,device)
    if history:
        plot_training_history(history,args.output_dir)
        (args.output_dir/'training_history.json').write_text(json.dumps(history,indent=2))
    if args.overfit_one:
        export_one_step(model,args,statistics,information,device,'train',prefix='overfit_train')
        print('Overfit diagnostic only: these are not held-out test results.')
        return
    export_one_step(model,args,statistics,information,device,'test')
    if args.evaluate_paper:
        if not information['has_paper_reference']:
            raise ValueError('paper_reference is absent from this dataset')
        export_one_step(model,args,statistics,information,device,'paper_reference',prefix='paper_reference')
    print(f'One-step outputs saved in {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
