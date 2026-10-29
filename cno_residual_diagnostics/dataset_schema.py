"""Full-state dataset inspection reused from the trained history CNO."""
import h5py
import numpy as np
NAMES = ("sxx","syy","sxy","vx","vy","p","qx","qy")

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


