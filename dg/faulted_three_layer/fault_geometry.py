"""Approved geometry. Public acquisition positions are (x, DEPTH), metres."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import brentq

WIDTH, HEIGHT = 1500., 1400.
SOURCE_DEPTH_M = np.array([250., 1200.])
RECEIVERS_DEPTH_M = np.array([[1000.,1000.],[1000.,600.],[1000.,200.]])
CONTACT_DEPTHS = (350., 700.)
TIP_TOP, TIP_BOTTOM = 200., 950.
MAX_THROW = 60.
SHALE_FRACTION = .30


@dataclass(frozen=True)
class Fault:
    name: str
    top_x: float
    dip_deg: float
    direction: int

    def x_at_depth(self, depth):
        return self.top_x+self.direction*(np.asarray(depth)-TIP_TOP)/np.tan(np.deg2rad(self.dip_deg))


FAULTS = (Fault('F1',25.,75.,1), Fault('F2',457.,75.,-1),
          Fault('F3',487.,75.,1), Fault('F4',919.,75.,-1), Fault('F5',949.,55.,1))


def depth_to_solver(points):
    p = np.asarray(points, float).copy()
    p[...,1] = HEIGHT-p[...,1]
    return p


def throw_at_depth(depth):
    z = np.asarray(depth)
    phase = np.clip((z-TIP_TOP)/(TIP_BOTTOM-TIP_TOP),0,1)
    return np.where((z>=TIP_TOP)&(z<=TIP_BOTTOM),MAX_THROW*np.sin(np.pi*phase)**2,0.)


def hanging_count(x, depth):
    x,z = np.broadcast_arrays(x,depth)
    count = np.zeros(x.shape,dtype=int)
    for f in FAULTS:
        is_hanging = x>f.x_at_depth(z) if f.direction==1 else x<f.x_at_depth(z)
        count += is_hanging & (z>=TIP_TOP) & (z<=TIP_BOTTOM)
    return count


def material_ids_depth(x, depth):
    x,z = np.broadcast_arrays(x,depth)
    restored = z-hanging_count(x,z)*throw_at_depth(z)
    return np.where(restored<CONTACT_DEPTHS[0],0,np.where(restored<CONTACT_DEPTHS[1],1,2)).astype(np.int64)


def material_ids_solver(points):
    points = np.asarray(points)
    return material_ids_depth(points[...,0], HEIGHT-points[...,1])


def configuration():
    return dict(width_m=WIDTH,height_m=HEIGHT,contact_depths_before_faulting_m=CONTACT_DEPTHS,
        fault_top_depth_m=TIP_TOP,fault_bottom_depth_m=TIP_BOTTOM,max_vertical_throw_per_fault_m=MAX_THROW,
        throw_taper='60*sin(pi*(depth-200)/750)^2 inside tips; zero outside',
        faults=[asdict(f) for f in FAULTS],shale_fraction=SHALE_FRACTION,
        source_depth_m=SOURCE_DEPTH_M.tolist(),receivers_depth_m=RECEIVERS_DEPTH_M.tolist(),
        source_solver_xy_m=depth_to_solver(SOURCE_DEPTH_M).tolist(),
        receivers_solver_xy_m=depth_to_solver(RECEIVERS_DEPTH_M).tolist())


def contact_segments():
    """Exact horizontal contact pieces, split at finite-fault traces.

    Maximum simultaneous hanging-wall count is three for this approved layout.
    Each count gives a unique root z-n*throw(z)=unfaulted_contact_depth.
    """
    segments=[]
    for base in CONTACT_DEPTHS:
        for n in range(4):
            z=brentq(lambda d:d-n*float(throw_at_depth(d))-base,0,HEIGHT,xtol=1e-10)
            cuts=sorted([0.,WIDTH]+[float(f.x_at_depth(z)) for f in FAULTS
                                    if TIP_TOP<z<TIP_BOTTOM and 0<f.x_at_depth(z)<WIDTH])
            for a,b in zip(cuts[:-1],cuts[1:]):
                if int(hanging_count((a+b)/2,z))==n:
                    segments.append(((a,z),(b,z)))
    return segments


def build_mesh(max_area=120., min_angle=28.):
    """Constrained quality triangulation, conforming to contacts and faults."""
    try:
        import triangle
    except ImportError as e:
        raise ImportError('Install the constrained mesher with: python -m pip install triangle') from e
    import aderdg_core as core
    if max_area<=0 or not 0<min_angle<=28:
        raise ValueError('max-area must be positive; min-angle must be in (0,28]')
    contacts=contact_segments()
    points=[]; index={}; segments=[]
    def vertex(p):
        key=tuple(np.round(p,8))
        if key not in index:
            index[key]=len(points);points.append(key)
        return index[key]
    def add(a,b):
        i,j=vertex(a),vertex(b)
        if i!=j:segments.append((i,j))
    for a,b in contacts:add(a,b)
    for f in FAULTS:
        zs=[TIP_TOP,TIP_BOTTOM]
        for a,b in contacts:
            for x,z in (a,b):
                if np.isclose(x,f.x_at_depth(z),atol=1e-7,rtol=0):zs.append(z)
        zs=np.unique(np.round(zs,8))
        for a,b in zip(zs[:-1],zs[1:]):add((f.x_at_depth(a),a),(f.x_at_depth(b),b))
    # Split rectangle edges where contacts end, preventing crossing PSLG segments.
    for x in (0.,WIDTH):
        zs=sorted(set([0.,HEIGHT]+[float(p[1]) for p in points if abs(p[0]-x)<1e-6]))
        for a,b in zip(zs[:-1],zs[1:]):add((x,a),(x,b))
    add((0,0),(WIDTH,0));add((0,HEIGHT),(WIDTH,HEIGHT))
    result=triangle.triangulate(dict(vertices=np.array(points),segments=np.array(segments)),
                                f'pq{min_angle:g}a{max_area:g}Q')
    vertices=depth_to_solver(result['vertices'])
    elements=result['triangles'].astype(np.int64)
    ev=vertices[elements]
    det=(ev[:,1,0]-ev[:,0,0])*(ev[:,2,1]-ev[:,0,1])-(ev[:,1,1]-ev[:,0,1])*(ev[:,2,0]-ev[:,0,0])
    swap=det<0
    elements[swap]=elements[swap][:,[0,2,1]]
    ev=vertices[elements]
    ids=material_ids_solver(ev.mean(1))
    for weights in ((.8,.1,.1),(.1,.8,.1),(.1,.1,.8)):
        check=material_ids_solver(np.einsum('j,eji->ei',weights,ev))
        if not np.array_equal(ids,check):
            raise RuntimeError('A triangle crosses a material contact; inspect the PSLG')
    neighbors=np.full((len(elements),3),-1,dtype=np.int64)
    faces=np.full_like(neighbors,-1);unmatched={}
    for e,tri in enumerate(elements):
        for f,(a,b) in enumerate(((0,1),(1,2),(2,0))):
            edge=tuple(sorted((int(tri[a]),int(tri[b]))))
            if edge in unmatched:
                other,of=unmatched.pop(edge)
                neighbors[e,f]=other;faces[e,f]=of;neighbors[other,of]=e;faces[other,of]=f
            else:unmatched[edge]=(e,f)
    # Every unmatched edge must belong to the exterior, not a crack in the mesh.
    for edge in unmatched:
        xy=vertices[list(edge)]
        if not any(np.allclose(xy[:,axis],value,atol=1e-6) for axis,value in
                   ((0,0.),(0,WIDTH),(1,0.),(1,HEIGHT))):
            raise RuntimeError('Mesh has an interior open edge')
    return core.TriangularMesh(vertices,elements,neighbors,faces,ids)
