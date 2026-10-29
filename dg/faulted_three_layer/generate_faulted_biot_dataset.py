"""Resumable eight-state, three-material faulted Biot ADER-DG dataset.

Default pilot: 8 train / 2 validation / 2 test. Use --preview-only first.
Acquisition CLI/configuration uses (x, DEPTH); arrays retain solver y-up order.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import time
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import aderdg_core as core
import fault_geometry as geo
import materials as rock
import interface_flux

NAMES=('sxx','syy','sxy','vx','vy','p','qx','qy')
UNITS=('Pa','Pa','Pa','m/s','m/s','Pa','m/s','m/s')
COLORS=['#7797a6','#b9b07a','#e7c58a']
SCHEMA='faulted_three_material_biot_v1'


def json_text(value):return json.dumps(value,sort_keys=True,allow_nan=False)
def strings(values):return np.asarray(values,dtype=h5py.string_dtype('utf-8'))


def source_integral(t,dt):
    # Exact antiderivative of (1-2*a^2)*exp(-a^2), a=pi*f*(t-delay).
    a=t-core.SOURCE_DELAY_S;b=a+dt;k=np.pi*core.SOURCE_FREQUENCY_HZ
    return b*np.exp(-(k*b)**2)-a*np.exp(-(k*a)**2)


def configure_core(materials):
    core.MATERIALS=tuple(materials)
    core.SOURCE_X_M,core.SOURCE_Y_M=geo.depth_to_solver(geo.SOURCE_DEPTH_M)


def preview(mesh,path):
    depth=geo.HEIGHT-mesh.vertices[:,1]
    fig,ax=plt.subplots(figsize=(11,9),constrained_layout=True)
    ax.tripcolor(mesh.vertices[:,0],depth,mesh.elements,facecolors=mesh.material_ids,
                 cmap=ListedColormap(COLORS),vmin=0,vmax=2,edgecolors='#555555',linewidth=.12)
    for f in geo.FAULTS:
        zz=np.array([geo.TIP_TOP,geo.TIP_BOTTOM])
        ax.plot(f.x_at_depth(zz),zz,color='#172e3e',lw=1.5)
    sx,sz=geo.SOURCE_DEPTH_M
    ax.scatter(sx,sz,marker='*',s=180,color='crimson',edgecolors='white',zorder=5)
    ax.annotate('50 Hz source (250, 1200)',(sx,sz),xytext=(8,12),textcoords='offset points',fontsize=10)
    for i,(x,z) in enumerate(geo.RECEIVERS_DEPTH_M,1):
        ax.scatter(x,z,marker='v',s=70,color='#146784',edgecolors='white',zorder=5)
        ax.annotate(f'R{i}',(x,z),xytext=(8,0),textcoords='offset points')
    ax.set(xlim=(0,geo.WIDTH),ylim=(geo.HEIGHT,0),xlabel='x (m)',ylabel='Depth below top (m)',
           title=f'Approved fault geometry on constrained DG mesh: {len(mesh.elements):,} triangles')
    ax.set_aspect('equal');fig.savefig(path,dpi=170);plt.close(fig)


def build_configuration(args,mesh,times):
    numerical_files=('aderdg_core.py','fault_geometry.py','materials.py','interface_flux.py','generate_faulted_biot_dataset.py')
    hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in numerical_files}
    return dict(schema_version=SCHEMA,geometry=geo.configuration(),source_frequency_hz=50.,
        source_delay_s=.04,source_strength=core.POINT_SOURCE_STRENGTH,
        counts=dict(train=args.train,validation=args.validation,test=args.test),
        nx=args.nx,ny=args.ny,times_s=times.tolist(),degree=args.degree,
        max_area_m2=args.max_area,min_angle_deg=args.min_angle,triangles=len(mesh.elements),
        mesh_hash=hashlib.sha256(mesh.vertices.tobytes()+mesh.elements.tobytes()).hexdigest(),
        dtype=args.dtype,dt_safety=args.dt_safety,seed=args.seed,compression=args.compression,
        material_mixing='0.3 shale / 0.7 sandstone: Hill Kdry, mu, Ks; arithmetic density/phi/tau; geometric permeability',
        baseline_materials=[asdict(rock.BASE_SHALE),asdict(rock.BASE_SANDSTONE)],
        interfaces='bonded open-pore characteristic coupling; no slip or fault gouge',
        same_material_flux='Rusanov, max adjacent speed',boundary='zero exterior trace, approximate absorbing numerical boundary; no PML',
        initial_condition='all eight states zero',code_sha256=hashes)


def create_file(path,config,mesh,x,y,times,ids,source):
    encoded=json_text(config)
    if path.exists():
        with h5py.File(path,'r') as f:
            if f.attrs.get('configuration_json')!=encoded:
                raise ValueError('Existing file configuration differs. Use a new filename or restore the original code/settings.')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'x') as f:
        f.attrs['configuration_json']=encoded;f.attrs['schema_version']=SCHEMA
        f.attrs['labels_are_numerical_ader_dg']=True;f.attrs['labels_are_analytical']=False
        coords=f.create_group('coords')
        for name,value in [('x_m',x),('y_m',y),('depth_m',geo.HEIGHT-y),('times_s',times)]:coords.create_dataset(name,data=value)
        meta=f.create_group('metadata')
        for name,value in [('input_names',rock.INPUT_NAMES),('output_names',NAMES),('output_units',UNITS),
                           ('matrix_layer_names',rock.MATERIAL_NAMES)]:meta.create_dataset(name,data=strings(value))
        meta.create_dataset('parameter_names',data=strings([f'{m}_{p}' for m in rock.MATERIAL_NAMES for p in rock.INPUT_NAMES[:-1]]))
        meta.attrs.update(source_frequency_hz=50.,source_delay_s=.04,source_strength=core.POINT_SOURCE_STRENGTH,
            source_x_m=250.,source_y_m=200.,source_depth_m=1200.,viscosity_pa_s=0.,
            frame_interval_s=float(times[1]-times[0]),coordinate_convention='Stored y is upward; depth=1400-y',
            state_equation='U_t + A U_x + B U_y = E U + S',
            fluid_variable='q = porosity*(fluid_velocity-solid_velocity), m/s',
            material_id_mapping='0 shale; 1 shaly_sandstone; 2 sandstone',
            boundary_condition=config['boundary'],interface_condition=config['interfaces'],
            source_map_role='Gaussian location descriptor only, not the DG forcing')
        meta.create_dataset('source_location_depth_m',data=geo.SOURCE_DEPTH_M)
        meta.create_dataset('receiver_locations_depth_m',data=geo.RECEIVERS_DEPTH_M)
        meta.create_dataset('receiver_locations_m',data=geo.depth_to_solver(geo.RECEIVERS_DEPTH_M))
        static=f.create_group('static')
        static.create_dataset('material_id',data=ids)
        X,Y=np.meshgrid(x,y)
        static.create_dataset('source_map',data=np.exp(-((X-250)**2+(Y-200)**2)/(2*35.**2)).astype('f4'))
        static.create_dataset('projected_source',data=source,compression='gzip')
        mg=f.create_group('mesh')
        for name,value in [('vertices_xy_m',mesh.vertices),('elements',mesh.elements),('material_ids',mesh.material_ids),
                           ('neighbors',mesh.neighbors),('neighbor_faces',mesh.neighbor_faces)]:mg.create_dataset(name,data=value,compression='gzip')
        compression=None if config['compression']=='none' else config['compression']
        opts={'compression':compression}
        if compression=='gzip':opts['compression_opts']=1
        for split,n in config['counts'].items():
            g=f.create_group(split)
            g.create_dataset('inputs',shape=(n,10,len(y),len(x)),dtype='f4',chunks=(1,10,len(y),len(x)),**opts)
            g.create_dataset('wavefields',shape=(n,len(times),8,len(y),len(x)),dtype='f4',chunks=(1,1,8,len(y),len(x)),**opts)
            g.create_dataset('receiver_wavefields',shape=(n,len(times),3,8),dtype='f4',chunks=(1,len(times),3,8),**opts)
            g.create_dataset('parameters',shape=(n,27),dtype='f8')
            for k in ('matrix_A','matrix_B','matrix_E'):g.create_dataset(k,shape=(n,3,8,8),dtype='f8')
            g.create_dataset('completed',data=np.zeros(n,dtype='u1'))
            g.create_dataset('runtime_s',data=np.full(n,np.nan))
            g.create_dataset('nominal_dt_s',data=np.full(n,np.nan))


def prepare(mesh,x,y,device,dtype,degree):
    mats=(rock.BASE_SHALE,rock.mixed(rock.BASE_SHALE,rock.BASE_SANDSTONE),rock.BASE_SANDSTONE)
    configure_core(mats)
    solver=core.GPUADERDG(mesh,degree,device,dtype)
    X,Y=np.meshgrid(x,y)
    sample=solver.make_sampler(np.c_[X.ravel(),Y.ravel()])
    ids=mesh.material_ids[sample.elements.cpu().numpy()].reshape(len(y),len(x))
    source=np.moveaxis(sample.sample(solver.source).reshape(len(y),len(x),8),-1,0).astype('f4')
    del sample,solver
    return ids,source


@torch.no_grad()
def simulate(mesh,x,y,times,mats,args,device,dtype):
    configure_core(mats)
    start=time.perf_counter()
    solver=core.GPUADERDG(mesh,args.degree,device,dtype)
    interface_faces=interface_flux.install(solver,mats)
    X,Y=np.meshgrid(x,y)
    grid=solver.make_sampler(np.c_[X.ravel(),Y.ravel()])
    receivers=solver.make_sampler(geo.depth_to_solver(geo.RECEIVERS_DEPTH_M))
    q=torch.zeros((len(mesh.elements),solver.nmodes,8),device=device,dtype=dtype)
    dt0=args.dt_safety*solver.cfl_diagnostic_s
    if not np.isfinite(dt0) or dt0<=0:raise ValueError('Invalid stable time step')
    out=np.zeros((len(times),8,len(y),len(x)),np.float32)
    traces=np.zeros((len(times),3,8),np.float32)
    print(f'  dt <= {dt0:.4g} s; about {int(np.ceil(times[-1]/dt0)):,} steps; {interface_faces} material-interface faces',flush=True)
    t=0.;steps=0
    for frame,target in enumerate(times[1:],1):
        while t<target-1e-13:
            dt=min(dt0,float(target-t))
            q=solver.step(q,dt,float(source_integral(t,dt)))
            t+=dt;steps+=1
        out[frame]=np.moveaxis(grid.sample(q).reshape(len(y),len(x),8),-1,0)
        traces[frame]=receivers.sample(q)
        if not np.isfinite(out[frame]).all() or not np.isfinite(traces[frame]).all():
            raise FloatingPointError(f'Nonfinite fields at t={target}; no completion flag will be written')
        if frame%max(1,(len(times)-1)//10)==0 or frame==len(times)-1:
            elapsed=time.perf_counter()-start
            print(f'  frame {frame}/{len(times)-1}, t={target:.4f}s, |Q|max={float(q.abs().max()):.3e}, ETA={elapsed/frame*(len(times)-1-frame)/60:.1f} min',flush=True)
    result=dict(runtime_s=time.perf_counter()-start,nominal_dt_s=dt0,steps=steps)
    del q,grid,receivers,solver
    return out,traces,result


def commit(path,split,index,inputs,fields,traces,mats,diagnostics):
    if not all(np.isfinite(v).all() for v in (inputs,fields,traces)):raise FloatingPointError('Nonfinite sample')
    with h5py.File(path,'r+') as f:
        g=f[split];g['completed'][index]=0
        g['inputs'][index]=inputs;g['wavefields'][index]=fields;g['receiver_wavefields'][index]=traces
        g['parameters'][index]=np.concatenate([rock.vector(m) for m in mats])
        for k,j in [('matrix_A',0),('matrix_B',1)]:g[k][index]=np.stack([m.matrices()[j] for m in mats])
        g['matrix_E'][index]=0.
        for k in ('runtime_s','nominal_dt_s'):g[k][index]=diagnostics[k]
        f.flush();g['completed'][index]=1;f.flush()


def plots(path,folder):
    folder.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'r') as f:
        if not f['train/completed'][0]:return
        times=f['coords/times_s'][:];x=f['coords/x_m'][:];y=f['coords/y_m'][:]
        for requested in (.1,.2,.25,.3,.4,.5):
            if requested>times[-1]+1e-10:continue
            idx=int(np.argmin(abs(times-requested)));fields=f['train/wavefields'][0,idx]
            fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
            for c,ax in enumerate(axes.flat):
                bound=max(float(abs(fields[c]).max()),1e-30)
                im=ax.imshow(fields[c,::-1,:],origin='upper',extent=[x[0],x[-1],geo.HEIGHT-y[0],geo.HEIGHT-y[-1]],
                             cmap='seismic',vmin=-bound,vmax=bound,aspect='equal')
                ax.set(title=NAMES[c],xlabel='x (m)',ylabel='Depth (m)');fig.colorbar(im,ax=ax,shrink=.8,label=UNITS[c])
            fig.suptitle(f'ADER-DG faulted model: training realization 0, t={times[idx]:.4f} s')
            fig.savefig(folder/f'train0_wavefields_t{times[idx]:.4f}.png',dpi=150);plt.close(fig)
        trace=f['train/receiver_wavefields'][0]
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
        for ax,c in zip(axes,[5,3,4]):
            for r in range(3):ax.plot(times,trace[:,r,c],label=f'R{r+1}')
            ax.set_ylabel(f'{NAMES[c]} ({UNITS[c]})');ax.grid(alpha=.25);ax.legend()
        axes[-1].set_xlabel('Time (s)');fig.savefig(folder/'train0_receiver_traces.png',dpi=160);plt.close(fig)


def verify(path):
    with h5py.File(path,'r') as f:
        if f.attrs.get('schema_version')!=SCHEMA:raise ValueError('Unexpected schema')
        for split in ('train','validation','test'):
            g=f[split];done=g['completed'][:]
            for i in np.flatnonzero(done):
                for k in ('inputs','wavefields','receiver_wavefields','parameters','matrix_A','matrix_B','matrix_E'):
                    if not np.isfinite(g[k][i]).all():raise ValueError(f'Nonfinite {split}/{i}/{k}')
            print(f'{split}: {int(done.sum())}/{len(done)} completed and finite',flush=True)


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('faulted_biot_8_2_2.h5'))
    p.add_argument('--train',type=int,default=8);p.add_argument('--validation',type=int,default=2);p.add_argument('--test',type=int,default=2)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--degree',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--max-area',type=float,default=120.);p.add_argument('--min-angle',type=float,default=28.)
    p.add_argument('--nx',type=int,default=128);p.add_argument('--ny',type=int,default=120)
    p.add_argument('--final-time',type=float,default=.5);p.add_argument('--frame-interval',type=float,default=.002)
    p.add_argument('--dt-safety',type=float,default=.9);p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--compression',choices=('gzip','lzf','none'),default='gzip')
    p.add_argument('--preview-only',action='store_true');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--skip-plots',action='store_true')
    a=p.parse_args()
    if min(a.train,a.validation,a.test)<1 or min(a.nx,a.ny)<5:p.error('Positive split sizes; nx/ny >= 5')
    if not all(np.isfinite(v) and v>0 for v in (a.max_area,a.min_angle,a.final_time,a.frame_interval,a.dt_safety)) or a.dt_safety>1 or a.min_angle>28:p.error('Invalid numerical settings')
    if a.preview_only and a.verify_only:p.error('Choose preview-only or verify-only')
    return a


def main():
    a=parse();a.output=a.output.expanduser()
    if a.verify_only:verify(a.output);return
    frames=int(round(a.final_time/a.frame_interval))
    if frames<4 or not np.isclose(frames*a.frame_interval,a.final_time,rtol=0,atol=1e-12):
        raise ValueError('Final time must be an integer multiple of frame interval, with >=5 frames')
    times=np.linspace(0,a.final_time,frames+1)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    dtype=getattr(torch,a.dtype)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=a.output.parent/(a.output.stem+'_figures');folder.mkdir(parents=True,exist_ok=True)
    print('Building the approved contact/fault-conforming mesh...',flush=True)
    mesh=geo.build_mesh(a.max_area,a.min_angle)
    preview(mesh,folder/'geometry_mesh.png')
    print(f'{len(mesh.elements):,} triangles; source (x,depth)={geo.SOURCE_DEPTH_M.tolist()}; solver (x,y)={geo.depth_to_solver(geo.SOURCE_DEPTH_M).tolist()}',flush=True)
    if a.preview_only:
        print(f'Preview saved to {folder}/geometry_mesh.png',flush=True);return
    x=np.linspace(0,geo.WIDTH,a.nx);y=np.linspace(0,geo.HEIGHT,a.ny)
    config=build_configuration(a,mesh,times)
    ids,source=prepare(mesh,x,y,device,dtype,a.degree)
    create_file(a.output,config,mesh,x,y,times,ids,source)
    counts=config['counts']
    print(f'Device={device}; {a.dtype}; {len(times)} frames; grid {a.ny}x{a.nx}; uncompressed wavefields ~{sum(counts.values())*len(times)*8*a.nx*a.ny*4/2**30:.1f} GiB',flush=True)
    for code,(split,n) in enumerate(counts.items()):
        with h5py.File(a.output,'r') as f:done=f[split+'/completed'][:]
        for index in range(n):
            if done[index]:continue
            print(f'{split} realization {index+1}/{n}',flush=True)
            mats=rock.sample(a.seed,code,index)
            fields,traces,diag=simulate(mesh,x,y,times,mats,a,device,dtype)
            commit(a.output,split,index,rock.fields(ids,mats),fields,traces,mats,diag)
            del fields,traces;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    if not a.skip_plots:plots(a.output,folder)
    print(f'Completed {a.output}. Rerun the SAME command to skip completed realizations.',flush=True)


if __name__=='__main__':main()
