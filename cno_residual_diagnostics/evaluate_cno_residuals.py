"""Post-training residual diagnostics for the original eight-field history CNO.

Loads existing PICNO or spectral-history checkpoint; no training, no new data.
Teacher-forced and predicted-sequence checks both use true-history one-step
predictions. The latter differences neighbouring predictions, not true states.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cno_model import SequentialCNO2d
from dataset_schema import inspect_data, NAMES
from biot_physics import stencil_mask
from residual_core import physical_terms, ResidualComparison

MAIN = [5,3,4]
PHYSICAL_UNITS = ('Pa/s','Pa/s','Pa/s','m/s^2','m/s^2','Pa/s','m/s^2','m/s^2')


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


@torch.inference_mode()
def predict_realization(model,truth,prop,source,times,stats,history,batch_size,amp):
    """Return physical one-step predictions; ONLY each query's true history enters."""
    device = truth.device
    mean = torch.tensor(stats['output_mean'],device=device,dtype=torch.float32)[None,:,None,None]
    std = torch.tensor(stats['output_std'],device=device,dtype=torch.float32)[None,:,None,None]
    im = torch.tensor(stats['input_mean'],device=device,dtype=torch.float32)[:,None,None]
    ins = torch.tensor(stats['input_std'],device=device,dtype=torch.float32)[:,None,None]
    prop = (prop-im)/ins
    nt,_,h,w = truth.shape
    predicted = torch.empty((nt-history,8,h,w),device=device,dtype=torch.float32)
    for start in range(history,nt,batch_size):
        end = min(start+batch_size,nt)
        frames = torch.stack([truth[j-history:j] for j in range(start,end)])
        frames = ((frames-mean)/std).reshape(end-start,8*history,h,w)
        time_maps = (times[start:end]/times[-1]).float()[:,None,None,None].expand(-1,1,h,w)
        inp = torch.cat((prop[None].expand(end-start,-1,-1,-1),
                         source[None,None].expand(end-start,1,-1,-1),time_maps,frames),dim=1)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp and device.type=='cuda'):
            correction = model(inp)
        # Match the saved trainer's float32 extrapolation and denormalization.
        normalized = 2*inp[:,-8:].float()-inp[:,-16:-8].float()+correction.float()
        predicted[start-history:end-history] = normalized*std+mean
    return predicted


def save_snapshot(out,mode,index,actual,terms_cno,terms_ref,comparison,meta,real):
    factor = torch.tensor(comparison.factor,device=terms_cno.device,dtype=terms_cno.dtype)[:,None,None]
    p = (terms_cno.sum(0)*factor).double().cpu().numpy()
    q = (terms_ref.sum(0)*factor).double().cpu().numpy()
    mask = comparison.valid.cpu().numpy()
    p[:,~mask],q[:,~mask] = np.nan,np.nan
    diff = p-q
    stem = f'{mode}_real{real}_residual_t{actual:.4f}'
    np.savez_compressed(out/f'{stem}.npz',cno=p,reference=q,difference=diff,valid=mask,
                        x_m=meta['x'][2:-2],z_m=meta['y'][2:-2],time_s=actual,index=index,output_names=np.array(NAMES),
                        scale_to_physical=comparison.unscale,physical_units=np.array(PHYSICAL_UNITS))
    fig,axes = plt.subplots(8,3,figsize=(14,23),constrained_layout=True)
    extent = [meta['x'][2],meta['x'][-3],meta['y'][2],meta['y'][-3]]
    cmap = plt.get_cmap('seismic').copy()
    cmap.set_bad('lightgray')
    for c,name in enumerate(NAMES):
        bound = max(float(np.nanpercentile(np.abs(np.concatenate((p[c][mask],q[c][mask]))),99.5)),1e-30)
        db = max(float(np.nanpercentile(abs(diff[c][mask]),99.5)),1e-30)
        for col,(field,limit,title) in enumerate(zip((q[c],p[c],diff[c]),(bound,bound,db),('ADER-DG','CNO','CNO minus reference'))):
            im = axes[c,col].imshow(field,origin='lower',extent=extent,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
            axes[c,col].set(title=f'R({name}): {title}',xlabel='x (m)',ylabel='z (m)')
            fig.colorbar(im,ax=axes[c,col],shrink=.8,label='Dimensionless residual')
    fig.suptitle(f'{mode}; realization {real}; t={actual:.4f}s. Gray = excluded. Color limits: 99.5th percentile.')
    fig.savefig(out/f'{stem}.png',dpi=130)
    plt.close(fig)


def save_time_outputs(out,mode,comparison,times):
    selected = comparison.counts>0
    indices = np.flatnonzero(selected)
    curves = {key:np.sqrt(values[selected]/comparison.counts[selected,None]) for key,values in comparison.sq.items()}
    np.savez_compressed(out/f'{mode}_residual_vs_time.npz',times_s=times[selected],indices=indices,
                        valid_pixel_counts=comparison.counts[selected],output_names=np.array(NAMES),**curves)
    fig,axes = plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for c,ax in enumerate(axes.flat):
        for key,label in (('reference','ADER-DG'),('cno','CNO'),('difference','Residual difference')):
            ax.plot(times[selected],curves[key][:,c],label=label)
        ax.set(title=NAMES[c],xlabel='Target time (s)',ylabel='Dimensionless residual RMS')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(out/f'{mode}_residual_vs_time.png',dpi=160)
    plt.close(fig)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('cno_residual_results'))
    p.add_argument('--split',choices=('train','validation','test'),default='test')
    p.add_argument('--mode',choices=('both','teacher','sequence'),default='both')
    p.add_argument('--realizations',type=int,default=0,help='0=all; positive number=first N for a pilot')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--residual-batch-size',type=int,default=8)
    p.add_argument('--residual-dtype',choices=('float64','float32'),default='float64')
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--no-amp',action='store_true',default=None,help='Disable model AMP; otherwise use checkpoint setting')
    p.add_argument('--physics-time-scale',type=float)
    p.add_argument('--boundary-cells',type=int)
    p.add_argument('--interface-cells',type=int)
    p.add_argument('--source-cells',type=int)
    p.add_argument('--plot-realization',type=int,default=0)
    p.add_argument('--plot-times',type=lambda s:[float(v) for v in s.split(',')],default=[.1,.2,.3,.4,.5])
    p.add_argument('--no-snapshot-plots',action='store_true')
    args = p.parse_args()
    if args.realizations<0 or min(args.batch_size,args.residual_batch_size)<=0 or args.plot_realization<0:
        p.error('Positive batch sizes and nonnegative realization counts required')
    if any(not np.isfinite(t) or t<0 for t in args.plot_times):
        p.error('Invalid plot times')
    return args


@torch.inference_mode()
def main():
    args = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float64 if args.residual_dtype=='float64' else torch.float32
    args.data,args.checkpoint,args.output_dir = args.data.expanduser(),args.checkpoint.expanduser(),args.output_dir.expanduser()
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('tag') not in ('full_state_biot_picno_v1','full_state_biot_cno_spectral_v1'):
        raise ValueError('Use the eight-field four-history PICNO/spectral checkpoint, not a direct hard-IC or three-field checkpoint')
    config,stats,saved_args = checkpoint['config'],checkpoint['statistics'],checkpoint.get('arguments',{})
    history = int(saved_args.get('history_frames',(config['input_channels']-12)//8))
    meta = inspect_data(args.data,history)
    if checkpoint['dataset_configuration']!=meta['configuration']:
        raise ValueError('Checkpoint and dataset configuration differ')
    if config['input_channels']!=len(meta['input_names'])+2+8*history or config['output_channels']!=8 or history<2:
        raise ValueError('Unsupported history/input/output configuration')
    for key,n in (('output_mean',8),('output_std',8),('input_mean',len(meta['input_names'])),('input_std',len(meta['input_names']))):
        values = np.asarray(stats[key])
        if values.shape!=(n,) or not np.isfinite(values).all() or ('std' in key and np.any(values<=0)):
            raise ValueError(f'Invalid checkpoint normalization: {key}')
    margins = {}
    for key,default in (('boundary_cells',4),('interface_cells',3),('source_cells',4)):
        value = getattr(args,key)
        margins[key] = int(saved_args.get(key,default) if value is None else value)
    time_scale = args.physics_time_scale if args.physics_time_scale is not None else float(saved_args.get('physics_time_scale',.02))
    if not np.isfinite(time_scale) or time_scale<=0:
        raise ValueError('Positive finite residual time scale required')
    no_amp = bool(saved_args.get('no_amp',False)) if args.no_amp is None else args.no_amp
    amp = device.type=='cuda' and not no_amp
    model = SequentialCNO2d(**config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    valid = stencil_mask(meta['layer_ids'],meta['projected_source'],**margins).to(device).bool()
    ids = torch.tensor(meta['layer_ids'],device=device,dtype=torch.long)
    source = torch.tensor(meta['projected_source'],device=device,dtype=dtype)[None]
    descriptor = torch.tensor(meta['source_map'],device=device,dtype=torch.float32)
    times = torch.tensor(meta['times'],device=device,dtype=dtype)
    nt = len(times)
    count = args.realizations or meta['counts'][args.split]
    if count>meta['counts'][args.split]:
        raise ValueError('Requested more realizations than available')
    modes = []
    if args.mode in ('both','teacher'):
        modes.append(('teacher_forced',history))
    if args.mode in ('both','sequence'):
        if nt<=history+2:
            raise ValueError('Need three predicted frames for sequence residuals')
        modes.append(('predicted_sequence',history+2))
    plot_indices = set()
    if not args.no_snapshot_plots:
        if args.plot_realization>=count:
            raise ValueError('Plot realization must be among evaluated realizations')
        for t in args.plot_times:
            if t>meta['times'][-1]+1e-9:
                raise ValueError('Plot time exceeds dataset interval')
            j = int(np.argmin(abs(meta['times']-t)))
            if j<max(start for _,start in modes):
                raise ValueError('Plot time precedes the first eligible residual target')
            plot_indices.add(j)
    comparisons = {name:ResidualComparison(nt,valid,stats['output_std'],time_scale) for name,_ in modes}
    fit_errors,fit_refs = np.zeros(8),np.zeros(8)
    started = time.perf_counter()
    print(f'Evaluating existing checkpoint epoch {checkpoint["epoch"]}; {count} {args.split} realizations; '
          f'history={history}; device={device}; residual dtype={args.residual_dtype}; model AMP={amp}',flush=True)
    with h5py.File(args.data,'r') as f:
        g = f[args.split]
        for real in range(count):
            fields = g['wavefields'][real]
            prop_np = g['inputs'][real]
            if not np.isfinite(fields).all() or not np.isfinite(prop_np).all():
                raise ValueError(f'Nonfinite data at {args.split}/{real}')
            if not np.array_equal(prop_np[meta['material_channel']],meta['layer_ids']):
                raise ValueError('Geometry differs from shared residual mask')
            truth = torch.tensor(fields,device=device,dtype=torch.float32)
            prop = torch.tensor(prop_np,device=device,dtype=torch.float32)
            prediction = predict_realization(model,truth,prop,descriptor,times,stats,history,args.batch_size,amp)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite model prediction')
            for start in range(history,nt,args.residual_batch_size):
                end = min(start+args.residual_batch_size,nt)
                p = prediction[start-history:end-history].double()
                q = truth[start:end].double()
                fit_errors += (p-q).square().sum((0,2,3)).cpu().numpy()
                fit_refs += q.square().sum((0,2,3)).cpu().numpy()
            matrices = [torch.tensor(g['matrix_'+key][real],device=device,dtype=dtype)[None] for key in ('A','B','E')]
            for name,first in modes:
                comp = comparisons[name]
                for start in range(first,nt,args.residual_batch_size):
                    end = min(start+args.residual_batch_size,nt)
                    indices = np.arange(start,end)
                    current = prediction[start-history:end-history].to(dtype)
                    previous = (truth[start-1:end-1] if name=='teacher_forced' else prediction[start-history-1:end-history-1]).to(dtype)
                    older = (truth[start-2:end-2] if name=='teacher_forced' else prediction[start-history-2:end-history-2]).to(dtype)
                    mats = [m.expand(end-start,-1,-1,-1) for m in matrices]
                    kwargs = dict(A=mats[0],B=mats[1],E=mats[2],ids=ids,source=source,times=times[start:end],
                                  dt=float(meta['dt']),dx=float(meta['dx']),dz=float(meta['dy']),
                                  frequency=float(meta['source_frequency_hz']),delay=float(meta['source_delay_s']))
                    cp = physical_terms(current,previous,older,**kwargs)
                    rp = physical_terms(truth[start:end].to(dtype),truth[start-1:end-1].to(dtype),truth[start-2:end-2].to(dtype),**kwargs)
                    comp.update(indices,cp,rp)
                    if real==args.plot_realization and plot_indices:
                        for j in sorted(plot_indices.intersection(indices.tolist())):
                            save_snapshot(out,name,j,float(meta['times'][j]),cp[j-start],rp[j-start],comp,meta,real)
            elapsed = time.perf_counter()-started
            print(f'  complete {real+1}/{count}, elapsed={elapsed/60:.1f} min, estimated remaining={elapsed/(real+1)*(count-real-1)/60:.1f} min',flush=True)
            del prediction,truth,prop
    fit_rel = np.sqrt(fit_errors/np.maximum(fit_refs,1e-100))
    summary = dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_epoch=checkpoint['epoch'],checkpoint_tag=checkpoint['tag'],
        trained_physics_loss_weight=saved_args.get('physics_loss_weight'),data=str(args.data.resolve()),split=args.split,
        evaluated_realizations=count,available_realizations=meta['counts'][args.split],history_frames=history,
        prediction_evaluation='true-history one-step; no recursive feedback',dt_saved_s=float(meta['dt']),
        physics_time_scale_s=time_scale,residual_dtype=args.residual_dtype,model_amp=amp,
        training_output_std_by_channel=dict(zip(NAMES,stats['output_std'])),
        physical_residual_units_by_channel=dict(zip(NAMES,PHYSICAL_UNITS)),mask_margins=margins,
        valid_fraction_of_full_grid=int(valid.sum())/meta['layer_ids'].size,
        prediction_relative_by_channel=dict(zip(NAMES,fit_rel.tolist())),
        prediction_relative_main=float(np.sqrt(np.mean(fit_rel[MAIN]**2))),prediction_relative_all=float(np.sqrt(np.mean(fit_rel**2))),
        residual_equation='R=Dt U + A Dx U + B Dz U - E U - S; BDF2 time, fourth-order centered space',
        residual_scaling='dimensionless residual = R * physics_time_scale_s / training_channel_std',
        relative_balance_definition='RMS(R) / [RMS(Ut)+RMS(A Ux)+RMS(B Uz)+RMS(E U)+RMS(S)], over the same cells/times',
        limitations=['Bulk masked grid diagnostic, not the original ADER-DG update.',
                     'Source, interface and exterior margins are excluded; IC/BC/interface constraints are not tested.',
                     'Near-zero residual is not proof of learned physics; reference consistency and nontrivial field accuracy also matter.',
                     'Predicted-sequence states are independently teacher-forced CNO outputs, not an autonomous rollout.'],
        modes={name:dict(first_target_time_s=float(meta['times'][first]),last_target_time_s=float(meta['times'][-1]),
                        target_examples=count*(nt-first),**comparisons[name].summary(NAMES)) for name,first in modes})
    write_json(out/'residual_summary.json',summary)
    with (out/'residual_by_channel.csv').open('w',newline='') as f:
        rows=[]
        for name,_ in modes:
            for channel,row in summary['modes'][name]['by_channel'].items():
                flat=dict(mode=name,channel=channel,physical_unit=PHYSICAL_UNITS[NAMES.index(channel)])
                for kind in ('cno','reference'):
                    flat.update({kind+'_'+k:v for k,v in row[kind].items()})
                flat.update({k:v for k,v in row.items() if k not in ('cno','reference')})
                rows.append(flat)
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for name,_ in modes:
        save_time_outputs(out,name,comparisons[name],meta['times'])
    mask_full=np.zeros(meta['layer_ids'].shape,bool)
    mask_full[2:-2,2:-2]=valid.cpu().numpy()
    np.savez_compressed(out/'physics_mask.npz',valid=mask_full,x_m=meta['x'],z_m=meta['y'])
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.imshow(mask_full,origin='lower',extent=[meta['x'][0],meta['x'][-1],meta['y'][0],meta['y'][-1]],cmap='gray',vmin=0,vmax=1,aspect='auto')
    ax.set(title='Bulk residual mask: white included, black excluded',xlabel='x (m)',ylabel='z (m)')
    fig.savefig(out/'physics_mask.png',dpi=160)
    plt.close(fig)
    print(f'Prediction all-field relative L2: {100*summary["prediction_relative_all"]:.4f}%',flush=True)
    for name,_ in modes:
        print(f'\n{name}: dimensionless residual RMS (CNO / reference / difference)',flush=True)
        for c,row in summary['modes'][name]['by_channel'].items():
            print(f"  {c}: {row['cno']['rms_dimensionless']:.6g} / {row['reference']['rms_dimensionless']:.6g} / {row['difference_rms_dimensionless']:.6g}",flush=True)
    print(f'Residual diagnostics saved in {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
