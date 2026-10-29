"""Plot all eight wavefields from a residual-only CNO checkpoint, without training.

Place this file alongside cno_model.py in your picno_residual_training folder.
Uses true history frames for each independent one-step prediction. Default:
test realization 0, target time 0.25 seconds. Optional --compare-reference adds
ADER-DG comparison plots; it never updates weights or computes a training loss.
"""
from pathlib import Path
import argparse
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cno_model import SequentialCNO2d

NAMES = ('sxx', 'syy', 'sxy', 'vx', 'vy', 'p', 'qx', 'qy')
UNITS = ('Pa', 'Pa', 'Pa', 'm/s', 'm/s', 'Pa', 'm/s', 'm/s')


def strings(values):
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in values)


def field_plot(fields, x, z, title, destination):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    extent = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    for c, ax in enumerate(axes.flat):
        limit = max(float(np.max(np.abs(fields[c]))), 1e-30)
        im = ax.imshow(fields[c], origin='lower', extent=extent, aspect='equal',
                       cmap='seismic', vmin=-limit, vmax=limit)
        ax.set_title(NAMES[c])
        ax.set_xlabel('x (m)'); ax.set_ylabel('vertical z (m)')
        fig.colorbar(im, ax=ax, shrink=.8, label=UNITS[c])
    fig.suptitle(title)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def comparison_plot(prediction, reference, x, z, title, destination):
    fig, axes = plt.subplots(8, 3, figsize=(13, 28), constrained_layout=True)
    extent = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    for c, row in enumerate(axes):
        difference = prediction[c]-reference[c]
        shared = max(float(np.max(np.abs(prediction[c]))), float(np.max(np.abs(reference[c]))), 1e-30)
        limits = (shared, shared, max(float(np.max(np.abs(difference))), 1e-30))
        for ax, field, limit, label in zip(row, (reference[c], prediction[c], difference), limits,
                                         ('ADER-DG', 'PICNO', 'PICNO - ADER-DG')):
            im = ax.imshow(field, origin='lower', extent=extent, aspect='equal',
                           cmap='seismic', vmin=-limit, vmax=limit)
            ax.set_title(f'{NAMES[c]}: {label}')
            ax.set_xlabel('x (m)'); ax.set_ylabel('vertical z (m)')
            fig.colorbar(im, ax=ax, shrink=.8, label=UNITS[c])
    fig.suptitle(title)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='best_residual.pt or final_residual.pt')
    p.add_argument('--output-dir', type=Path, default=Path('residual_wavefield_snapshots'))
    p.add_argument('--plot-times', type=lambda s: [float(v) for v in s.split(',')], default=[.25])
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    p.add_argument('--realization', type=int, default=0, help='Zero-based sample index within selected split')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--no-amp', action='store_true', help='Disable AMP even if used in training')
    p.add_argument('--compare-reference', action='store_true', help='Also save eight-state reference/difference plots')
    args = p.parse_args()
    if args.realization < 0 or not args.plot_times or not np.isfinite(args.plot_times).all():
        p.error('Use a nonnegative realization and finite snapshot times')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    checkpoint = torch.load(args.checkpoint.expanduser(), map_location='cpu', weights_only=False)
    if checkpoint.get('tag') != 'biot_residual_only_history_v1':
        raise ValueError('Select best_residual.pt or final_residual.pt from the residual-only run')
    config, stats = checkpoint['config'], checkpoint['statistics']
    history = int(checkpoint['arguments']['history_frames'])
    if config['output_channels'] != 8 or history < 2:
        raise ValueError('Checkpoint must predict all eight states with at least two history frames')
    for key in ('input_mean', 'input_std', 'output_mean', 'output_std'):
        if not np.isfinite(stats[key]).all() or (key.endswith('std') and np.any(np.asarray(stats[key]) <= 0)):
            raise ValueError('Invalid checkpoint normalization')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = SequentialCNO2d(**config).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    amp = bool(checkpoint['arguments'].get('amp', False)) and not args.no_amp and device.type == 'cuda'
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    epoch = int(checkpoint['epoch'])
    print(f'Checkpoint epoch {epoch}; {history} true history frames; device={device}; no training', flush=True)
    with h5py.File(args.data.expanduser(), 'r') as f:
        if str(f.attrs.get('configuration_json', '')) != checkpoint['dataset_configuration']:
            raise ValueError('Checkpoint and dataset configurations differ')
        if strings(f['metadata/output_names'][:]) != NAMES:
            raise ValueError('Dataset must contain the expected eight fields')
        if len(f['metadata/input_names'])+2+8*history != config['input_channels']:
            raise ValueError('Dataset input channels do not match the model')
        g = f[args.split]
        real = args.realization
        if real >= len(g['completed']) or g['completed'][real] != 1:
            raise ValueError('Selected realization does not exist or is incomplete')
        times, x, z = f['coords/times_s'][:], f['coords/x_m'][:], f['coords/y_m'][:]
        source_map = f['static/source_map'][:]
        prop = g['inputs'][real]
        im = np.asarray(stats['input_mean'], np.float32)[:, None, None]
        ins = np.asarray(stats['input_std'], np.float32)[:, None, None]
        om = np.asarray(stats['output_mean'], np.float32)[:, None, None]
        osd = np.asarray(stats['output_std'], np.float32)[:, None, None]
        saved = set()
        for requested in args.plot_times:
            if requested < times[0]-1e-10 or requested > times[-1]+1e-10:
                raise ValueError(f'Requested time {requested} is outside saved time range')
            idx = int(np.argmin(np.abs(times-requested)))
            if idx < history:
                raise ValueError(f'Need {history} true history frames; earliest target is {times[history]:.6g} s')
            if idx in saved:
                continue
            saved.add(idx)
            actual = float(times[idx])
            q = g['wavefields'][real, idx-history:idx]
            q = (q-om[None])/osd[None]
            time_map = np.full(q.shape[-2:], actual/float(times[-1]), np.float32)
            inputs = np.concatenate(((prop-im)/ins, source_map[None], time_map[None],
                                     q.reshape(-1, *q.shape[-2:]))).astype(np.float32)
            if not np.isfinite(inputs).all():
                raise ValueError('Nonfinite model inputs')
            inp = torch.from_numpy(inputs)[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                correction = model(inp)
            predicted = 2*inp[:, -8:].float()-inp[:, -16:-8].float()+correction.float()
            prediction = predicted[0].cpu().numpy()*osd+om
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite prediction')
            stem = f'{args.split}_real{real}_epoch{epoch}_wavefields_t{actual:.4f}'
            title = f'PICNO wavefields | checkpoint epoch {epoch} | {args.split} realization {real} | t={actual:.4f} s'
            field_plot(prediction, x, z, title, out/f'{stem}.png')
            arrays = dict(prediction=prediction, output_names=np.array(NAMES), units=np.array(UNITS),
                          x_m=x, z_m=z, time_s=actual, requested_time_s=requested,
                          input_history_times_s=times[idx-history:idx], checkpoint_epoch=epoch,
                          checkpoint_path=str(args.checkpoint.expanduser().resolve()),
                          prediction_mode='true-history one-step; no recursive feedback')
            if args.compare_reference:
                reference = g['wavefields'][real, idx]
                arrays.update(reference=reference, difference=prediction-reference)
                comparison_plot(prediction, reference, x, z, title, out/f'{stem}_comparison.png')
            np.savez_compressed(out/f'{stem}.npz', **arrays)
            print(f'Requested {requested:.6g} s -> saved {actual:.6g} s; history={times[idx-history:idx].tolist()}', flush=True)
            print(f'Saved {out / (stem+".png")}', flush=True)


if __name__ == '__main__':
    main()
