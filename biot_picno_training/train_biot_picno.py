"""Four-frame, one-step full-state CNO with optional discrete Biot regularization.

No predicted state is fed back into the input. Supervised baseline available
with --physics-loss-weight 0. Run --calibrate-only before physical training.
Bulk residuals use physical units, saved matrices and source, BDF2 time and
fourth-order spatial differences. Optional interface and incoming boundary
penalties use approximate traces; this is not the original DG discrete operator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from cno_model import SequentialCNO2d, wavefield_balanced_loss, spatial_gradient_loss, wave_energy_loss
from biot_physics import BiotResidual, incoming_matrices

NAMES = ('sxx', 'syy', 'sxy', 'vx', 'vy', 'p', 'qx', 'qy')
MAIN = [5, 3, 4]
TAG = 'full_state_biot_picno_v1'


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def save_checkpoint(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def strings(values):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in values]


def inspect_data(path, history):
    with h5py.File(path, 'r') as f:
        if tuple(strings(f['metadata/output_names'][:])) != NAMES:
            raise ValueError(f'Expected fields {NAMES}; use the eight-field PICNO dataset')
        t = f['coords/times_s'][:]
        x, y = f['coords/x_m'][:], f['coords/y_m'][:]
        for values, label in ((t, 'time'), (x, 'x'), (y, 'y')):
            if len(values) < 5 or not np.all(np.diff(values) > 0) or not np.allclose(
                    np.diff(values), values[1]-values[0], rtol=1e-8, atol=1e-10):
                raise ValueError(f'{label} coordinates must be uniformly increasing')
        if len(t) <= history:
            raise ValueError('Insufficient frames for requested history')
        input_names = strings(f['metadata/input_names'][:])
        material_channel = input_names.index('material_id')
        ids = f['train/inputs'][0, material_channel]
        if not np.isin(ids, (0, 1)).all():
            raise ValueError('Expected layer IDs 0=shale,1=sandstone')
        meta = dict(input_names=input_names, material_channel=material_channel,
                    x=x, y=y, times=t, dt=t[1]-t[0], dx=x[1]-x[0], dy=y[1]-y[0],
                    layer_ids=ids.astype(np.int64), source_map=f['static/source_map'][:],
                    projected_source=f['static/projected_source'][:], counts={})
        for key in ('source_frequency_hz', 'source_delay_s', 'interface_y_m'):
            meta[key] = float(f['metadata'].attrs[key])
        if float(f['metadata'].attrs.get('viscosity_pa_s', 0)) != 0:
            raise ValueError('This package targets the supplied inviscid dataset')
        for split in ('train', 'validation', 'test'):
            g = f[split]
            n = len(g['completed'])
            if n == 0 or not np.all(g['completed'][:] == 1):
                raise ValueError(f'{split} contains incomplete or missing realizations')
            if g['wavefields'].shape != (n, len(t), 8, len(y), len(x)):
                raise ValueError(f'Invalid shape in {split}')
            if g['inputs'].shape != (n, len(input_names), len(y), len(x)):
                raise ValueError(f'Invalid inputs in {split}')
            for name in ('matrix_A', 'matrix_B', 'matrix_E'):
                if g[name].shape != (n, 2, 8, 8) or not np.isfinite(g[name][:]).all():
                    raise ValueError(f'Invalid {name} in {split}')
            if np.any(g['matrix_E'][:] != 0):
                raise ValueError('Inviscid generator should store zero E')
            meta['counts'][split] = n
        if meta['projected_source'].shape != (8, len(y), len(x)):
            raise ValueError('Invalid projected source')
        if not np.isfinite(meta['projected_source']).all():
            raise ValueError('Nonfinite source')
        meta['configuration'] = str(f.attrs.get('configuration_json', ''))
    print(f"Grid {len(y)}x{len(x)}, {len(t)} frames, dt_saved={meta['dt']:.6g} s, "
          f"splits {meta['counts']}", flush=True)
    return meta


def fingerprint(path, meta, stride):
    st = path.stat()
    content = [str(path.resolve()), st.st_size, st.st_mtime_ns,
               meta['configuration'], stride]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def training_statistics(path, meta, args, out):
    token = fingerprint(path, meta, args.statistics_frame_stride)
    cache = out/'normalization.json'
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get('fingerprint') == token:
            print('Using cached training-only normalization', flush=True)
            return data
    c = len(meta['input_names'])
    input_sum, input_sq = np.zeros(c), np.zeros(c)
    field_sum, field_sq = np.zeros(8), np.zeros(8)
    count_input, count_output = 0, 0
    print('Computing training-only statistics (streaming HDF5; first run may take time)', flush=True)
    with h5py.File(path, 'r') as f:
        for i in range(meta['counts']['train']):
            a = f['train/inputs'][i].astype(np.float64)
            if not np.isfinite(a).all():
                raise ValueError(f'Nonfinite training inputs at realization {i}')
            input_sum += a.sum(axis=(1, 2))
            input_sq += (a*a).sum(axis=(1, 2))
            count_input += a.shape[1]*a.shape[2]
            for j in range(0, len(meta['times']), args.statistics_frame_stride):
                q = f['train/wavefields'][i, j].astype(np.float64)
                if not np.isfinite(q).all():
                    raise ValueError(f'Nonfinite training state {i}/{j}')
                field_sum += q.sum(axis=(1, 2))
                field_sq += (q*q).sum(axis=(1, 2))
                count_output += q.shape[1]*q.shape[2]
            if i % 20 == 0 or i+1 == meta['counts']['train']:
                print(f"  statistics {i+1}/{meta['counts']['train']}", flush=True)
    def moments(total, square, count, outputs=False):
        mean = total/count
        std = np.sqrt(np.maximum(square/count-mean*mean, 0))
        if outputs and np.any(std <= 1e-30):
            raise ValueError('An output field has zero variance; check generation')
        std = np.where(std <= 1e-12*np.maximum(np.abs(mean), 1), 1., std) if not outputs else std
        return mean.tolist(), std.tolist()
    im, ins = moments(input_sum, input_sq, count_input)
    om, osd = moments(field_sum, field_sq, count_output, True)
    stats = dict(input_mean=im, input_std=ins, output_mean=om, output_std=osd,
                 fingerprint=token, frame_stride=args.statistics_frame_stride)
    write_json(cache, stats)
    return stats


class HistoryDataset(Dataset):
    def __init__(self, path, split, meta, stats, args):
        self.path, self.split, self.meta = str(path), split, meta
        self.history, self.windows = args.history_frames, len(meta['times'])-args.history_frames
        self.count = meta['counts'][split]
        self.im = np.asarray(stats['input_mean'], np.float32)[:, None, None]
        self.istd = np.asarray(stats['input_std'], np.float32)[:, None, None]
        self.om = np.asarray(stats['output_mean'], np.float32)[None, :, None, None]
        self.ostd = np.asarray(stats['output_std'], np.float32)[None, :, None, None]
        self.handle = None
        with h5py.File(path, 'r') as f:
            self.A = f[f'{split}/matrix_A'][:].astype(np.float32)
            self.B = f[f'{split}/matrix_B'][:].astype(np.float32)
            self.E = f[f'{split}/matrix_E'][:].astype(np.float32)
        self.incoming = None
        if args.boundary_loss_weight > 0:
            scale = np.asarray(stats['output_std'])
            self.incoming = np.stack([np.stack([
                incoming_matrices(-a, scale), incoming_matrices(a, scale),
                incoming_matrices(-b, scale), incoming_matrices(b, scale)])
                for a, b in zip(self.A.astype(np.float64), self.B.astype(np.float64))])

    def __len__(self):
        return self.count*self.windows

    def __getstate__(self):
        data = self.__dict__.copy()
        data['handle'] = None
        return data

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, 'r')
        real, offset = divmod(int(index), self.windows)
        target_index = self.history+offset
        g = self.handle[self.split]
        prop = g['inputs'][real]
        if not np.isfinite(prop).all():
            raise ValueError(f'Nonfinite {self.split} material map at realization {real}')
        if not np.array_equal(prop[self.meta['material_channel']], self.meta['layer_ids']):
            raise ValueError('Material geometry changed; shared stencil mask is invalid')
        q = g['wavefields'][real, target_index-self.history:target_index+1]
        if not np.isfinite(q).all():
            raise ValueError(f'Nonfinite {self.split} state {real}/{target_index}')
        q = (q-self.om)/self.ostd
        history = q[:-1].reshape(-1, *q.shape[-2:])
        time_map = np.full(q.shape[-2:], self.meta['times'][target_index]/self.meta['times'][-1], np.float32)
        inp = np.concatenate(((prop-self.im)/self.istd,
              self.meta['source_map'][None], time_map[None], history)).astype(np.float32)
        data = dict(inputs=inp, target=q[-1], A=self.A[real], B=self.B[real], E=self.E[real],
                    time=np.float32(self.meta['times'][target_index]), index=target_index, realization=real)
        if self.incoming is not None:
            data['incoming'] = self.incoming[real]
        return data


def sampled_indices(dataset, limit, seed):
    if limit == 0 or limit >= dataset.windows:
        return None
    rng = np.random.default_rng(seed)
    return np.concatenate([i*dataset.windows+rng.choice(dataset.windows, limit, replace=False)
                           for i in range(dataset.count)]).tolist()


def loader(dataset, args, training=False, seed=0):
    limit = args.windows_per_realization if training else args.validation_windows
    indices = sampled_indices(dataset, limit, seed)
    selected = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(selected, batch_size=args.batch_size, shuffle=training,
                      num_workers=args.num_workers, pin_memory=args.device.type == 'cuda',
                      generator=torch.Generator().manual_seed(seed))


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict(model, inputs, amp):
    with torch.autocast(device_type=inputs.device.type, dtype=torch.float16,
                        enabled=amp and inputs.device.type == 'cuda'):
        correction = model(inputs)
    baseline = 2*inputs[:, -8:].float()-inputs[:, -16:-8].float()
    return baseline+correction.float(), baseline


def loss_components(pred, batch, physics, args, weight):
    target, inp = batch['target'], batch['inputs']
    field = wavefield_balanced_loss(pred, target)
    grad = spatial_gradient_loss(pred, target)
    amplitude = wave_energy_loss(pred, target)
    bulk = pred.new_zeros(())
    if weight > 0:
        residual = physics.residual(pred, inp[:, -8:], inp[:, -16:-8],
                        batch['A'], batch['B'], batch['E'], batch['time'])
        bulk = physics.bulk_loss(residual)
    interface = physics.interface_loss(pred) if args.interface_loss_weight else pred.new_zeros(())
    boundary = physics.boundary_loss(pred, batch['incoming']) if args.boundary_loss_weight else pred.new_zeros(())
    total = field+args.gradient_loss_weight*grad+args.amplitude_loss_weight*amplitude
    total = total+weight*bulk+args.interface_loss_weight*interface+args.boundary_loss_weight*boundary
    values = dict(total=total, supervised=field, gradient=grad, amplitude=amplitude,
                  physics=bulk, interface=interface, boundary=boundary)
    if not all(torch.isfinite(v).item() for v in values.values()):
        raise FloatingPointError('Nonfinite loss; inspect data and residual scales')
    return values


def epoch_pass(model, batches, physics, stats, args, weight, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    count, sums = 0, {}
    error, reference, baseline_error = np.zeros(8), np.zeros(8), np.zeros(8)
    for batch in batches:
        batch = move(batch, args.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred, baseline = predict(model, batch['inputs'], not args.no_amp)
            values = loss_components(pred, batch, physics, args, weight)
            if training:
                scaler.scale(values['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                              error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
        n = len(pred)
        count += n
        for key, value in values.items():
            sums[key] = sums.get(key, 0)+float(value.detach())*n
        if not training:
            p, target, b = [physics.physical(v).double() for v in (pred, batch['target'], baseline)]
            axes = (0, 2, 3)
            error += (p-target).square().sum(axes).cpu().numpy()
            baseline_error += (b-target).square().sum(axes).cpu().numpy()
            reference += target.square().sum(axes).cpu().numpy()
    result = {k: v/count for k, v in sums.items()}
    if not training:
        rel = np.sqrt(error/np.maximum(reference, 1e-100))
        result['relative_by_channel'] = rel.tolist()
        result['relative_main'] = float(np.sqrt(np.mean(rel[MAIN]**2)))
        result['relative_all'] = float(np.sqrt(np.mean(rel**2)))
        result['baseline_relative_by_channel'] = np.sqrt(baseline_error/np.maximum(reference, 1e-100)).tolist()
    return result


@torch.no_grad()
def calibrate(dataset, physics, args, out):
    n = min(args.calibration_pairs, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), n, replace=False))
    batches = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size,
                         num_workers=args.num_workers)
    residual_sq = np.zeros(8)
    baseline_sq = np.zeros(8)
    num_pixels = 0
    interface_sum = boundary_sum = count = 0
    for batch in batches:
        batch = move(batch, args.device)
        target, inp = batch['target'], batch['inputs']
        base = 2*inp[:, -8:]-inp[:, -16:-8]
        for predicted, accum in ((target, residual_sq), (base, baseline_sq)):
            r = physics.residual(predicted, inp[:, -8:], inp[:, -16:-8],
                              batch['A'], batch['B'], batch['E'], batch['time'])
            accum += (r.double().square()*physics.valid[None, None]).sum((0, 2, 3)).cpu().numpy()
        num_pixels += len(target)*float(physics.valid.sum())
        if args.interface_loss_weight:
            interface_sum += float(physics.interface_loss(target))*len(target)
        if args.boundary_loss_weight:
            boundary_sum += float(physics.boundary_loss(target, batch['incoming']))*len(target)
        count += len(target)
    result = dict(pairs=count, split='train', derivative='BDF2_time_fourth_order_space',
                  time_scale_s=physics.time_scale,
                  valid_fraction=float(physics.valid.sum())/(physics.ids.numel()),
                  reference_residual_rms_by_channel=dict(zip(NAMES, np.sqrt(residual_sq/num_pixels).tolist())),
                  extrapolation_residual_rms_by_channel=dict(zip(NAMES, np.sqrt(baseline_sq/num_pixels).tolist())),
                  reference_interface_loss=interface_sum/count if args.interface_loss_weight else None,
                  reference_boundary_loss=boundary_sum/count if args.boundary_loss_weight else None,
                  note='Dimensionless discrete residuals; not prediction percentages or a proof of PDE accuracy')
    write_json(out/'reference_residual_calibration.json', result)
    print(json.dumps(result, indent=2), flush=True)
    plt.figure(figsize=(10, 4))
    at = np.arange(8)
    plt.bar(at-.2, np.sqrt(residual_sq/num_pixels), .4, label='ADER-DG reference')
    plt.bar(at+.2, np.sqrt(baseline_sq/num_pixels), .4, label='Linear extrapolation')
    plt.xticks(at, NAMES)
    plt.ylabel('Dimensionless residual RMS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/'reference_residual_calibration.png', dpi=160)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.imshow(physics.valid.cpu(), origin='lower', cmap='gray', vmin=0, vmax=1)
    plt.title('Bulk Biot residual cells: white included, black excluded')
    plt.tight_layout()
    plt.savefig(out/'physics_mask.png', dpi=160)
    plt.close()
    return result


def history_plot(history, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    e = [r['epoch'] for r in history]
    for phase in ('train', 'validation'):
        axes[0].plot(e, [r[phase]['supervised'] for r in history], label=phase)
        axes[1].plot(e, [r[phase]['physics'] for r in history], label=phase)
    axes[0].set_title('Supervised field loss')
    axes[1].set_title('Unweighted Biot residual loss')
    axes[2].plot(e, [r['validation']['relative_main'] for r in history], label='p, vx, vy')
    axes[2].set_title('Validation one-step relative L2')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'training_history.png', dpi=170)
    plt.close(fig)


def train(model, train_ds, val_ds, physics, stats, meta, args, out, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=8)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp and args.device.type == 'cuda')
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and args.device.type == 'cuda')
    history, best, stale, first_epoch = [], float('inf'), 0, 1
    if args.resume:
        previous = torch.load(args.resume.expanduser(), map_location=args.device, weights_only=False)
        if previous.get('tag') != TAG or previous['config'] != config:
            raise ValueError('Resume checkpoint architecture does not match')
        if previous['dataset_configuration'] != meta['configuration'] or previous['statistics'] != stats:
            raise ValueError('Resume checkpoint dataset/normalization does not match')
        invariant = ['physics_loss_weight', 'physics_time_scale', 'physics_warmup_epochs',
                     'interface_loss_weight', 'boundary_loss_weight', 'gradient_loss_weight',
                     'amplitude_loss_weight', 'boundary_cells', 'interface_cells', 'source_cells',
                     'history_frames', 'windows_per_realization', 'validation_windows', 'seed']
        if any(previous['arguments'][k] != getattr(args, k) for k in invariant):
            raise ValueError('Keep sampling and physics/loss settings unchanged when resuming')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        history, best = previous['history'], previous['best_relative_main']
        stale, first_epoch = previous['stale_epochs'], previous['epoch']+1
        # Keep an inherited best model if resuming in a new results folder.
        previous_best = args.resume.expanduser().parent/'best_picno.pt'
        if not (out/'best_picno.pt').exists() and previous_best.exists():
            import shutil
            shutil.copy2(previous_best, out/'best_picno.pt')
        print(f'Resuming training at epoch {first_epoch}', flush=True)
    val_loader = loader(val_ds, args, seed=args.seed)
    for epoch in range(first_epoch, args.epochs+1):
        started = time.perf_counter()
        weight = args.physics_loss_weight
        if args.physics_warmup_epochs:
            weight *= min(1., epoch/args.physics_warmup_epochs)
        training = epoch_pass(model, loader(train_ds, args, True, args.seed+epoch), physics,
                              stats, args, weight, optimizer, scaler)
        validation = epoch_pass(model, val_loader, physics, stats, args, weight)
        metric = validation['relative_main']
        scheduler.step(metric)
        record = dict(epoch=epoch, train=training, validation=validation,
                      physics_weight=weight, learning_rate=optimizer.param_groups[0]['lr'])
        history.append(record)
        improved = metric < best-args.min_delta
        if improved:
            best, stale = metric, 0
        else:
            stale += 1
        payload = dict(tag=TAG, model=model.state_dict(), config=config, statistics=stats,
                       epoch=epoch, history=history, best_relative_main=best,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), stale_epochs=stale,
                       selection='validation one-step relative L2 of p,vx,vy',
                       arguments={k: str(v) if isinstance(v, (Path, torch.device)) else v
                                  for k, v in vars(args).items()},
                       dataset_configuration=meta['configuration'])
        if improved:
            save_checkpoint(out/'best_picno.pt', payload)
        save_checkpoint(out/'final_picno.pt', payload)
        write_json(out/'training_history.json', history)
        print(f"Epoch {epoch}/{args.epochs}: field={training['supervised']:.5g}, "
              f"Biot={training['physics']:.5g}, val main L2={metric:.5g}, "
              f"physics weight={weight:.4g}, {time.perf_counter()-started:.1f}s", flush=True)
        if not args.no_early_stopping and stale >= args.patience:
            print('Early stopping', flush=True)
            break
    history_plot(history, out)


@torch.no_grad()
def evaluate_test(model, dataset, physics, args, out, prefix='test'):
    # Always ALL test windows, regardless of validation sampling setting.
    batches = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=args.device.type == 'cuda')
    nt = len(dataset.meta['times'])
    errors, refs, baselines = np.zeros((nt, 8)), np.zeros((nt, 8)), np.zeros((nt, 8))
    model.eval()
    for batch in batches:
        batch = move(batch, args.device)
        p, b = predict(model, batch['inputs'], not args.no_amp)
        p, b, q = [physics.physical(v).double() for v in (p, b, batch['target'])]
        er = (p-q).square().sum((2, 3)).cpu().numpy()
        br = (b-q).square().sum((2, 3)).cpu().numpy()
        tr = q.square().sum((2, 3)).cpu().numpy()
        indices = batch['index'].cpu().numpy()
        np.add.at(errors, indices, er)
        np.add.at(baselines, indices, br)
        np.add.at(refs, indices, tr)
    relative = np.sqrt(errors.sum(0)/np.maximum(refs.sum(0), 1e-100))
    base_rel = np.sqrt(baselines.sum(0)/np.maximum(refs.sum(0), 1e-100))
    metrics = dict(evaluation='four-true-history one-step; no recursive feedback',
                   history_frames=args.history_frames, realizations=dataset.count,
                   relative_by_channel=dict(zip(NAMES, relative.tolist())),
                   baseline_relative_by_channel=dict(zip(NAMES, base_rel.tolist())),
                   relative_main=float(np.sqrt(np.mean(relative[MAIN]**2))),
                   relative_all=float(np.sqrt(np.mean(relative**2))))
    write_json(out/f'{prefix}_one_step_metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    per_time = np.sqrt(errors/np.maximum(refs, 1e-100))
    np.savez_compressed(out/f'{prefix}_error_vs_time.npz', times=dataset.meta['times'],
                        relative=per_time, baseline_relative=np.sqrt(baselines/np.maximum(refs, 1e-100)),
                        output_names=np.array(NAMES), first_predicted_index=args.history_frames)
    fig, ax = plt.subplots(figsize=(9, 4))
    h = args.history_frames
    for c in MAIN:
        ax.plot(dataset.meta['times'][h:], per_time[h:, c], label=NAMES[c])
    ax.set_xlabel('Target time (s)')
    ax.set_ylabel('One-step relative L2')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/f'{prefix}_one_step_error_vs_time.png', dpi=170)
    plt.close(fig)
    for requested in args.plot_times:
        idx = int(np.argmin(np.abs(dataset.meta['times']-requested)))
        if idx < h:
            raise ValueError(f'Plot time {requested} is an input seed frame; choose >= {dataset.meta["times"][h]}')
        record = dataset[idx-h]  # realization zero, true history
        inp = torch.tensor(record['inputs'], device=args.device)[None]
        prediction, baseline = predict(model, inp, not args.no_amp)
        prediction = physics.physical(prediction)[0].cpu().numpy()
        truth = physics.physical(torch.tensor(record['target'], device=args.device)[None])[0].cpu().numpy()
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        extent = [dataset.meta['x'][0], dataset.meta['x'][-1], dataset.meta['y'][0], dataset.meta['y'][-1]]
        for row, c in enumerate(MAIN):
            diff = prediction[c]-truth[c]
            bound = max(float(np.percentile(np.abs(truth[c]), 99.5)), 1e-30)
            db = max(float(np.percentile(np.abs(diff), 99.5)), 1e-30)
            for col, (field, limit, title) in enumerate(zip(
                    (truth[c], prediction[c], diff), (bound, bound, db), ('ADER-DG', 'CNO', 'CNO - ADER-DG'))):
                im = axes[row, col].imshow(field, origin='lower', extent=extent, cmap='seismic',
                                          vmin=-limit, vmax=limit, aspect='auto')
                axes[row, col].set_title(f'{NAMES[c]}: {title}')
                axes[row, col].set_xlabel('x (m)')
                axes[row, col].set_ylabel('y / vertical z (m)')
                fig.colorbar(im, ax=axes[row, col], shrink=.8, label='Pa' if c==5 else 'm/s')
        actual = dataset.meta['times'][idx]
        fig.suptitle(f'One-step prediction, realization 0, t={actual:.4f}s')
        fig.tight_layout()
        fig.savefig(out/f'{prefix}_sample0_comparison_t{actual:.4f}.png', dpi=170)
        plt.close(fig)
        np.savez_compressed(out/f'{prefix}_sample0_t{actual:.4f}.npz', prediction=prediction,
                            reference=truth, time_s=actual, output_names=np.array(NAMES))


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('picno_results'))
    p.add_argument('--history-frames', type=int, choices=(2, 4), default=4)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--layers', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--padding', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--windows-per-realization', type=int, default=16)
    p.add_argument('--validation-windows', type=int, default=0, help='0=all; otherwise fixed sampled windows per realization')
    p.add_argument('--physics-loss-weight', type=float, default=0.01)
    p.add_argument('--physics-time-scale', type=float, default=0.02, help='Residual scale time in seconds, default one 50-Hz period')
    p.add_argument('--physics-warmup-epochs', type=int, default=0, help='Optional weight ramp; no multi-step curriculum')
    p.add_argument('--interface-loss-weight', type=float, default=0, help='Optional open-pore bonded interface penalty')
    p.add_argument('--boundary-loss-weight', type=float, default=0, help='Optional zero-incoming-characteristic penalty')
    p.add_argument('--gradient-loss-weight', type=float, default=.1)
    p.add_argument('--amplitude-loss-weight', type=float, default=.1)
    p.add_argument('--boundary-cells', type=int, default=4)
    p.add_argument('--interface-cells', type=int, default=3)
    p.add_argument('--source-cells', type=int, default=4)
    p.add_argument('--statistics-frame-stride', type=int, default=1, help='1=all training frames; larger=sampling for normalization')
    p.add_argument('--calibration-pairs', type=int, default=64)
    p.add_argument('--calibrate-only', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path, help='Resume training from final_picno.pt with unchanged settings')
    p.add_argument('--evaluate-final', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-early-stopping', action='store_true')
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--min-delta', type=float, default=1e-6)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--plot-times', type=lambda s: [float(x) for x in s.split(',')], default=[.1, .2, .3, .4, .5])
    args = p.parse_args()
    positive = ['epochs', 'batch_size', 'width', 'learning_rate', 'statistics_frame_stride', 'calibration_pairs', 'patience']
    nonnegative = ['windows_per_realization', 'validation_windows', 'physics_loss_weight', 'physics_warmup_epochs',
                   'interface_loss_weight', 'boundary_loss_weight', 'gradient_loss_weight', 'amplitude_loss_weight',
                   'num_workers', 'padding', 'min_delta']
    if any(getattr(args, k) <= 0 for k in positive) or any(getattr(args, k) < 0 for k in nonnegative):
        p.error('Counts/rates must be positive and loss weights/margins nonnegative')
    if args.calibrate_only and args.evaluate_only:
        p.error('Choose only one of calibrate-only and evaluate-only')
    if args.resume and (args.calibrate_only or args.evaluate_only):
        p.error('Resume is only for training')
    if any(not np.isfinite(t) or t < 0 for t in args.plot_times):
        p.error('Invalid plot times')
    return args


def main():
    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    args.data, args.output_dir = args.data.expanduser(), args.output_dir.expanduser()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f'Device: {args.device}', flush=True)
    meta = inspect_data(args.data, args.history_frames)
    config = dict(input_channels=len(meta['input_names'])+2+8*args.history_frames,
                  output_channels=8, width=args.width, layers=args.layers, padding=args.padding)
    checkpoint = None
    if args.evaluate_only:
        path = (args.checkpoint or out/'best_picno.pt').expanduser()
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        if checkpoint.get('tag') != TAG:
            raise ValueError('Checkpoint does not belong to this full-state PICNO trainer')
        if checkpoint['config']['input_channels'] != config['input_channels']:
            raise ValueError('Checkpoint history/input channels do not match')
        if checkpoint['dataset_configuration'] != meta['configuration']:
            raise ValueError('Checkpoint and dataset configuration differ')
        config, stats = checkpoint['config'], checkpoint['statistics']
    else:
        stats = training_statistics(args.data, meta, args, out)
    physics = BiotResidual(meta, stats, args).to(args.device)
    datasets = {split: HistoryDataset(args.data, split, meta, stats, args)
                for split in ('train', 'validation', 'test')}
    try:
        if not args.evaluate_only:
            calibrate(datasets['train'], physics, args, out)
            if args.calibrate_only:
                print(f'Calibration saved to {out}; inspect it before physics training', flush=True)
                return
        model = SequentialCNO2d(**config).to(args.device)
        print(f'CNO input channels={config["input_channels"]}, outputs=8, parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
        if args.evaluate_only:
            model.load_state_dict(checkpoint['model'])
        else:
            train(model, datasets['train'], datasets['validation'], physics, stats, meta, args, out, config)
            path = out/('final_picno.pt' if args.evaluate_final else 'best_picno.pt')
            if not path.exists():
                raise ValueError('No selected checkpoint in output folder; resume using its existing results folder')
            checkpoint = torch.load(path, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
        print(f'Evaluating checkpoint epoch {checkpoint["epoch"]}', flush=True)
        evaluate_test(model, datasets['test'], physics, args, out)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == '__main__':
    main()
