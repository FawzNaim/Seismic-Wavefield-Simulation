"""Unchanged supervised objective functions from the CNO comparison run."""
import torch
import torch.nn.functional as F

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
    """Compare normalized field RMS amplitudes; not physical energy conservation."""
    predicted_rms = torch.sqrt(torch.mean(predictions.float() ** 2, dim=(2, 3)) + 1.0e-8)
    target_rms = torch.sqrt(torch.mean(targets.float() ** 2, dim=(2, 3)) + 1.0e-8)
    valid = (target_rms >= minimum_target_rms).float()
    loss = F.smooth_l1_loss(
        predicted_rms, target_rms, beta=0.10, reduction="none"
    )
    return torch.sum(loss * valid) / torch.clamp(torch.sum(valid), min=1.0)


