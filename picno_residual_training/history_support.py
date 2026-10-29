"""Dataset, normalization, and model-input utilities from the original trainer.
No supervised loss or model-selection logic is used here.
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


