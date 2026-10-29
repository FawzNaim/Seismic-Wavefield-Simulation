"""Residual-only, four-true-history Biot CNO. No target-error objective."""
from pathlib import Path
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import history_support as base
from biot_physics import BiotResidual, spatial_derivatives, apply_layer_matrices
from cno_model import SequentialCNO2d

TAG = 'biot_residual_only_history_v1'
NAMES = base.NAMES


class HistoryOnlyDataset(base.HistoryDataset):
    """Read the four input frames only; never read a target wavefield."""
    def __getitem__(self, index):
        if getattr(self, 'owner_pid', None) != os.getpid():
            self.close()
            self.owner_pid = os.getpid()
        if self.handle is None:
            import h5py
            self.handle = h5py.File(self.path, 'r')
        real, offset = divmod(int(index), self.windows)
        target_index = self.history+offset
        g = self.handle[self.split]
        prop = g['inputs'][real]
        if not np.isfinite(prop).all() or not np.array_equal(prop[self.meta['material_channel']], self.meta['layer_ids']):
            raise ValueError('Nonfinite material map or unsupported changing layer geometry')
        q = g['wavefields'][real, target_index-self.history:target_index]
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite input history')
        q = (q-self.om)/self.ostd
        time_map = np.full(q.shape[-2:], self.meta['times'][target_index]/self.meta['times'][-1], np.float32)
        inp = np.concatenate(((prop-self.im)/self.istd, self.meta['source_map'][None],
                              time_map[None], q.reshape(-1, *q.shape[-2:]))).astype(np.float32)
        data = dict(inputs=inp, A=self.A[real], B=self.B[real], E=self.E[real],
                    time=np.float64(self.meta['times'][target_index]))
        if self.incoming is not None:
            data['incoming'] = self.incoming[real]
        return data


class ResidualPhysics(BiotResidual):
    def residual(self, predicted, previous, older, A, B, E, target_times):
        # Explicit precision outside network autocast; gradients reach the network.
        dtype = self.residual_dtype
        mean, std = self.mean.to(dtype), self.std.to(dtype)
        u, prev, old = [v.to(dtype)*std+mean for v in (predicted, previous, older)]
        ut = (3*u-4*prev+old)/(2*self.dt)
        ux, uy = spatial_derivatives(u, self.dx, self.dy)
        ids = self.ids[2:-2, 2:-2]
        center = u[..., 2:-2, 2:-2]
        phase = torch.pi*self.frequency*(target_times.to(dtype)-self.delay)
        ricker = (1-2*phase.square())*torch.exp(-phase.square())
        source = self.source.to(dtype)[..., 2:-2, 2:-2]*ricker[:, None, None, None]
        r = ut[..., 2:-2, 2:-2]+apply_layer_matrices(A.to(dtype), ux, ids)
        r = r+apply_layer_matrices(B.to(dtype), uy, ids)
        r = r-apply_layer_matrices(E.to(dtype), center, ids)-source
        return r*(self.time_scale/std)


def summarize(square_sums, count):
    if count <= 0:
        raise ValueError('No residual samples')
    mse = np.asarray(square_sums, dtype=np.float64)/count
    rms = np.sqrt(mse)
    return dict(rms_by_channel=dict(zip(NAMES, rms.tolist())),
                mean_rms=float(rms.mean()), mean_square=float(mse.mean()),
                pooled_rms=float(np.sqrt(mse.mean())), values_per_channel=int(count))


def residual_pass(model, batches, physics, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    squares = torch.zeros(8, device=args.device, dtype=torch.float64)
    count, examples, objective_sum = 0, 0, 0.
    valid_count = int(physics.valid.sum().item())
    with torch.set_grad_enabled(training):
        for batch in batches:
            batch = base.move(batch, args.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred, _ = base.predict(model, batch['inputs'], args.amp)
            residual = physics.residual(pred, batch['inputs'][:, -8:],
                batch['inputs'][:, -16:-8], batch['A'], batch['B'], batch['E'], batch['time'])
            bulk = physics.bulk_loss(residual)
            objective = args.physics_loss_weight*bulk
            if args.interface_loss_weight:
                objective = objective+args.interface_loss_weight*physics.interface_loss(pred)
            if args.boundary_loss_weight:
                objective = objective+args.boundary_loss_weight*physics.boundary_loss(pred, batch['incoming'])
            if not torch.isfinite(objective).item():
                raise FloatingPointError('Nonfinite objective')
            if training:
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip,
                                              error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            n = len(pred)
            squares += (residual.detach().double().square()*physics.valid[None, None]).sum((0, 2, 3))
            count += n*valid_count
            examples += n
            objective_sum += float(objective.detach())*n
    result = summarize(squares.cpu().numpy(), count)
    result['objective'] = objective_sum/examples
    result['pairs'] = examples
    return result


def export_history(history, out):
    base.write_json(out/'residual_history.json', history)
    columns = ['epoch', 'learning_rate', 'seconds', 'train_online_mean_rms',
               'validation_mean_rms', 'validation_pooled_rms', 'validation_mean_square',
               'train_pairs', 'validation_pairs', 'best_so_far', 'stale_epochs']
    columns += [f'{phase}_{c}_rms' for phase in ('train_online', 'validation') for c in NAMES]
    tmp = out/'residual_history.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in history:
            tr, va = row['train_online'], row['validation']
            flat = {k: row[k] for k in ('epoch', 'learning_rate', 'seconds', 'best_so_far', 'stale_epochs')}
            flat.update(train_online_mean_rms=None if tr is None else tr['mean_rms'],
                        validation_mean_rms=va['mean_rms'], validation_pooled_rms=va['pooled_rms'],
                        validation_mean_square=va['mean_square'],
                        train_pairs=0 if tr is None else tr['pairs'], validation_pairs=va['pairs'])
            for phase, values in (('train_online', tr), ('validation', va)):
                flat.update({f'{phase}_{c}_rms': None if values is None else values['rms_by_channel'][c]
                             for c in NAMES})
            writer.writerow(flat)
    tmp.replace(out/'residual_history.csv')
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    epochs = [r['epoch'] for r in history]
    axes.flat[0].plot(epochs, [r['validation']['mean_rms'] for r in history], label='validation mean RMS')
    axes.flat[0].set_title('Checkpoint / stopping metric')
    for ax, c in zip(list(axes.flat)[1:], NAMES):
        ax.plot(epochs, [r['validation']['rms_by_channel'][c] for r in history], label='validation')
        training = [r for r in history if r['train_online'] is not None]
        ax.plot([r['epoch'] for r in training],
                [r['train_online']['rms_by_channel'][c] for r in training], label='train (online)', alpha=.7)
        ax.set_title(c)
    for ax in axes.flat:
        ax.set_xlabel('epoch'); ax.set_ylabel('dimensionless RMS residual')
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'residual_history.png', dpi=150); plt.close(fig)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('picno_residual_results'))
    initial = p.add_mutually_exclusive_group()
    initial.add_argument('--initialize-from', type=Path, help='Load model and statistics; start a new 200-epoch fine-tuning run')
    initial.add_argument('--resume', type=Path, help='Continue this trainer in the SAME results directory')
    p.add_argument('--epochs', type=int, default=200, help='Maximum total epochs in this run, excluding initialization epoch 0')
    p.add_argument('--history-frames', type=int, choices=(2, 4), default=4)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--layers', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--padding', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--physics-loss-weight', type=float, default=.01)
    p.add_argument('--physics-time-scale', type=float, default=None)
    p.add_argument('--residual-dtype', choices=('float32', 'float64'), default='float64')
    p.add_argument('--interface-loss-weight', type=float, default=0.)
    p.add_argument('--boundary-loss-weight', type=float, default=0.)
    for margin in ('boundary', 'interface', 'source'):
        p.add_argument(f'--{margin}-cells', type=int, default=None)
    p.add_argument('--windows-per-realization', type=int, default=16)
    p.add_argument('--validation-windows', type=int, default=0, help='0=all target times; positive=fixed subset per realization')
    p.add_argument('--statistics-frame-stride', type=int, default=1)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr-patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=1e-5, help='Absolute improvement in mean validation RMS required to reset patience')
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--amp', action='store_true', help='Optional network float16; residual arithmetic keeps requested precision')
    p.add_argument('--no-early-stopping', action='store_true')
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = p.parse_args()
    for k in ('epochs', 'width', 'batch_size', 'learning_rate', 'physics_loss_weight',
              'patience', 'lr_patience', 'statistics_frame_stride', 'gradient_clip'):
        if not np.isfinite(getattr(args, k)) or getattr(args, k) <= 0:
            p.error(f'{k} must be positive and finite')
    for k in ('num_workers', 'windows_per_realization', 'validation_windows', 'padding',
              'min_delta', 'interface_loss_weight', 'boundary_loss_weight'):
        if not np.isfinite(getattr(args, k)) or getattr(args, k) < 0:
            p.error(f'{k} must be finite and nonnegative')
    return args


def main():
    args = parse()
    args.data, args.output_dir = args.data.expanduser(), args.output_dir.expanduser()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'residual_history.json').exists() and not args.resume:
        raise ValueError('Existing run: use --resume or a new output directory')
    if args.resume and args.resume.expanduser().resolve().parent != out.resolve():
        raise ValueError('Resume in the original output directory to preserve its best checkpoint')
    args.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = None
    if args.initialize_from or args.resume:
        checkpoint = torch.load((args.initialize_from or args.resume).expanduser(), map_location='cpu', weights_only=False)
        allowed = (TAG,) if args.resume else (TAG, base.TAG, 'full_state_biot_cno_spectral_v1')
        if checkpoint.get('tag') not in allowed:
            raise ValueError('Unsupported checkpoint mapping; requires the original true-history model')
        args.history_frames = int(checkpoint['arguments']['history_frames'])
    saved_args = {} if checkpoint is None else checkpoint['arguments']
    for k, default in (('physics_time_scale', .02), ('boundary_cells', 4), ('interface_cells', 3), ('source_cells', 4)):
        if getattr(args, k) is None:
            setattr(args, k, saved_args.get(k, default))
    if not np.isfinite(args.physics_time_scale) or args.physics_time_scale <= 0:
        raise ValueError('physics-time-scale must be finite and positive')
    meta = base.inspect_data(args.data, args.history_frames)
    config = dict(input_channels=len(meta['input_names'])+2+8*args.history_frames,
                  output_channels=8, width=args.width, layers=args.layers, padding=args.padding)
    if checkpoint is not None:
        if checkpoint['dataset_configuration'] != meta['configuration']:
            raise ValueError('Checkpoint and dataset configurations differ')
        config, stats = checkpoint['config'], checkpoint['statistics']
        if config['input_channels'] != len(meta['input_names'])+2+8*args.history_frames or config['output_channels'] != 8:
            raise ValueError('Invalid checkpoint mapping')
        print(f"Loaded architecture and statistics from checkpoint epoch {checkpoint['epoch']}", flush=True)
    else:
        stats = base.training_statistics(args.data, meta, args, out)
    for key in ('input_std', 'output_std'):
        if not np.isfinite(stats[key]).all() or np.any(np.asarray(stats[key]) <= 0):
            raise ValueError('Invalid normalization scales')
    for key in ('input_mean', 'output_mean'):
        if not np.isfinite(stats[key]).all():
            raise ValueError('Invalid normalization means')
    base.write_json(out/'normalization.json', stats)
    physics = ResidualPhysics(meta, stats, args).to(args.device)
    physics.residual_dtype = getattr(torch, args.residual_dtype)
    model = SequentialCNO2d(**config).to(args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience,
                                                        threshold=args.min_delta, threshold_mode='abs')
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and args.device.type == 'cuda')
    arguments = {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()}
    datasets = {s: HistoryOnlyDataset(args.data, s, meta, stats, args) for s in ('train', 'validation', 'test')}
    # Close parent handles before worker creation. Each loader owns its HDF5 handles.
    history, best, anchor, stale, first = [], float('inf'), float('inf'), 0, 1
    best_epoch = 0
    if args.resume:
        invariants = ('history_frames', 'windows_per_realization', 'validation_windows', 'seed',
            'physics_loss_weight', 'physics_time_scale', 'residual_dtype', 'interface_loss_weight',
            'boundary_loss_weight', 'boundary_cells', 'interface_cells', 'source_cells', 'amp',
            'min_delta', 'patience', 'lr_patience', 'gradient_clip')
        if any(saved_args[k] != arguments[k] for k in invariants):
            raise ValueError('Keep residual, sampling, precision, and stopping settings unchanged on resume')
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, anchor, stale, first, best_epoch = (checkpoint[k] for k in
            ('history', 'best_mean_rms', 'patience_anchor', 'stale_epochs', 'next_epoch', 'best_epoch'))
        if not (out/'best_residual.pt').exists():
            raise ValueError('Missing best_residual.pt in resume folder')
    def payload(epoch):
        return dict(tag=TAG, model=model.state_dict(), config=config, statistics=stats,
            epoch=epoch, next_epoch=epoch+1, history=history, best_mean_rms=best, best_epoch=best_epoch,
            patience_anchor=anchor, stale_epochs=stale, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), arguments=arguments,
            dataset_configuration=meta['configuration'], selection='minimum arithmetic mean of 8 validation dimensionless RMS residuals',
            initialization_checkpoint=None if args.initialize_from is None else str(args.initialize_from),
            residual_mode='teacher_forced; true previous/older states; predicted current state')
    try:
        validation_loader = base.loader(datasets['validation'], args, seed=args.seed)
        print(f'Device={args.device}; residual={args.residual_dtype}; valid fraction={float(physics.valid.sum())/physics.ids.numel():.4f}', flush=True)
        if not args.resume:
            initial = residual_pass(model, validation_loader, physics, args)
            best = anchor = initial['mean_rms']
            history.append(dict(epoch=0, train_online=None, validation=initial, learning_rate=args.learning_rate,
                                seconds=0., best_so_far=True, stale_epochs=0))
            base.save_checkpoint(out/'best_residual.pt', payload(0))
            base.save_checkpoint(out/'final_residual.pt', payload(0))
            export_history(history, out)
            print(f'Initialization: validation mean RMS={best:.8g}', flush=True)
        for epoch in range(first, args.epochs+1):
            if not args.no_early_stopping and stale >= args.patience:
                print('Plateau patience already exhausted; stopping', flush=True)
                break
            started = time.perf_counter()
            training = residual_pass(model, base.loader(datasets['train'], args, True, args.seed+epoch),
                                     physics, args, optimizer, scaler)
            validation = residual_pass(model, validation_loader, physics, args)
            metric = validation['mean_rms']
            scheduler.step(metric)
            improved = metric < best
            if improved:
                best, best_epoch = metric, epoch
            if metric < anchor-args.min_delta:
                anchor, stale = metric, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_online=training, validation=validation,
                learning_rate=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started,
                best_so_far=improved, stale_epochs=stale))
            if improved:
                base.save_checkpoint(out/'best_residual.pt', payload(epoch))
            base.save_checkpoint(out/'final_residual.pt', payload(epoch))
            export_history(history, out)
            print(f"Epoch {epoch}/{args.epochs}: validation mean RMS={metric:.8g}, best={best:.8g} "
                  f"at epoch {best_epoch}; stale={stale}/{args.patience}; {history[-1]['seconds']:.1f}s", flush=True)
            print('  '+', '.join(f'{c}={validation["rms_by_channel"][c]:.6g}' for c in NAMES), flush=True)
        chosen = torch.load(out/'best_residual.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(chosen['model'])
        # Test is consulted once after selection; never drives stopping.
        test = residual_pass(model, base.loader(datasets['test'], args, seed=args.seed), physics, args)
        summary = dict(best_epoch=chosen['epoch'], best_validation=chosen['history'][-1]['validation'],
            best_checkpoint_test_residual=test, last_epoch=history[-1]['epoch'],
            stopped_for_plateau=not args.no_early_stopping and stale >= args.patience,
            residual_mode='teacher_forced', objective='physics only; no target matching',
            warning='Masked finite-difference residual, not native DG residual or proof of source/IC/BC correctness',
            train_metric='online while weights change', validation_metric='fixed samples, frozen end-of-epoch weights',
            valid_fraction=float(physics.valid.sum())/physics.ids.numel(),
            physics_time_scale_s=args.physics_time_scale, residual_dtype=args.residual_dtype)
        base.write_json(out/'residual_summary.json', summary)
        print(f'Finished. Best residual checkpoint: epoch {chosen["epoch"]}, validation mean RMS={best:.8g}', flush=True)
    finally:
        for ds in datasets.values():
            ds.close()


if __name__ == '__main__':
    main()
