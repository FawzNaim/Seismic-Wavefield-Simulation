"""Bonded open-pore Biot transmission at contacts; no fault slip dynamics.

Continuity: two total-traction components, two solid-velocity components,
pore pressure, and normal relative-fluid flux. Three incoming characteristics
on either side determine six unknown amplitudes. Same-material faces retain
the original Rusanov-type flux; exterior faces retain zero exterior traces.
"""
import numpy as np
import torch

SCALES=np.array([1e7,1e7,1e7,1.,1.,1e7,1.,1.])


def continuity_matrix(normal):
    nx,ny=normal
    C=np.zeros((6,8))
    C[0,0]=nx;C[0,2]=ny
    C[1,2]=nx;C[1,1]=ny
    C[2,3]=1;C[3,4]=1;C[4,5]=1
    C[5,6]=nx;C[5,7]=ny
    return C


def coupled_flux(material_left,material_right,normal):
    def normal_matrix(m):
        A,B=m.matrices();return normal[0]*A+normal[1]*B
    AL,AR=normal_matrix(material_left),normal_matrix(material_right)
    L=AL*SCALES[None,:]/SCALES[:,None]
    R=AR*SCALES[None,:]/SCALES[:,None]
    def modes(matrix,sign):
        speeds,vectors=np.linalg.eig(matrix)
        if np.max(np.abs(speeds.imag))>1e-6:
            raise ValueError('Non-real characteristic decomposition')
        selected=sign*speeds.real>max(abs(speeds.real))*1e-7
        if selected.sum()!=3:
            raise ValueError('Expected three propagating incoming Biot modes per side')
        if np.max(np.abs(vectors[:,selected].imag))>1e-6:
            raise ValueError('Non-real propagating characteristic vectors')
        return vectors.real[:,selected]
    Vminus,Vplus=modes(L,-1),modes(R,1)
    C=continuity_matrix(normal)
    D=np.concatenate((C@Vminus,-C@Vplus),axis=1)
    if np.linalg.cond(D)>1e10:
        raise ValueError('Ill-conditioned open-pore interface solve')
    amplitudes=np.linalg.solve(D,C)
    JL=Vminus@amplitudes[:3];JR=Vplus@amplitudes[3:]
    I=np.eye(8)
    # Normalized trace maps: UL*=(I-JL)UL+JL UR; UR*=-JR UL+(I+JR)UR.
    maps=(L@(I-JL),L@JL,-R@(I+JR),R@JR)
    physical=tuple(m*SCALES[:,None]/SCALES[None,:] for m in maps)
    return physical,(I-JL,JL,-JR,I+JR)


def install(solver,materials):
    mesh=solver.mesh
    left=[];lf=[];right=[];rf=[];pl=[];ml=[];pr=[];mr=[]
    edge=((0,1),(1,2),(2,0))
    for e in range(len(mesh.elements)):
        for f in range(3):
            other=mesh.neighbors[e,f]
            if other<0 or other<=e or mesh.material_ids[e]==mesh.material_ids[other]:continue
            a,b=edge[f]
            tangent=mesh.element_vertices[e,b]-mesh.element_vertices[e,a]
            normal=np.array([tangent[1],-tangent[0]])/np.linalg.norm(tangent)
            maps,_=coupled_flux(materials[mesh.material_ids[e]],materials[mesh.material_ids[other]],normal)
            left.append(e);lf.append(f);right.append(other);rf.append(mesh.neighbor_faces[e,f])
            for values,matrix in zip((pl,ml,pr,mr),maps):values.append(2*matrix)
    if left:
        for target,e,f,values in ((solver.face_plus,left,lf,pl),(solver.face_minus,left,lf,ml),
                                   (solver.face_plus,right,rf,pr),(solver.face_minus,right,rf,mr)):
            target[torch.tensor(e,device=solver.device),torch.tensor(f,device=solver.device)]=torch.as_tensor(
                np.stack(values),device=solver.device,dtype=solver.dtype)
    return len(left)
