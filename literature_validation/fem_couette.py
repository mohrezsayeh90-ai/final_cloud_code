"""Independent Taylor--Hood FEM for the equilibrium free-spin Couette problem.

The angular velocity is a global DOF. A Galerkin constraint maps circle velocity
DOFs to rigid rotation; the associated virtual-work equation enforces zero torque.
Published angular velocities are never used by this solver.
"""
from dataclasses import asdict
import json
from pathlib import Path
import math
import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri, Basis, ElementTriP2, ElementTriP1, ElementVector, BilinearForm, asm
from skfem.helpers import ddot, sym_grad, dot, mul, grad, div
import triangle


def make_mesh(spacing=.06, circle_points=96):
    nx,ny=int(round(2/spacing)),int(round(1/spacing))
    # Identical node spacing on left/right. Y forbids independent boundary splits.
    outer=np.vstack([np.column_stack([np.linspace(0,2,nx+1)[:-1],np.zeros(nx)]),
                     np.column_stack([np.full(ny,2.),np.linspace(0,1,ny+1)[:-1]]),
                     np.column_stack([np.linspace(2,0,nx+1)[:-1],np.ones(nx)]),
                     np.column_stack([np.zeros(ny),np.linspace(1,0,ny+1)[:-1]])])
    angle=np.arange(circle_points)*2*np.pi/circle_points
    circle=np.column_stack([1+.125*np.cos(angle),.5+.125*np.sin(angle)])
    vertices=np.vstack([outer,circle]);no=len(outer)
    segments=np.vstack([np.column_stack([np.arange(no),np.roll(np.arange(no),-1)]),
                        np.column_stack([no+np.arange(circle_points),no+np.roll(np.arange(circle_points),-1)])])
    triangulation=triangle.triangulate(dict(vertices=vertices,segments=segments,holes=np.array([[1.,.5]])),f'pq28a{spacing**2/2:.10f}Y')
    mesh=MeshTri(triangulation['vertices'].T,triangulation['triangles'].T)
    return mesh


def constraints(bu,bp,mesh):
    nu,np_=bu.N,bp.N
    coords=bu.doflocs
    component=np.zeros(nu,dtype=int)
    component[bu.split_indices()[1]]=1
    wall=np.isclose(coords[1],0,atol=1e-10)|np.isclose(coords[1],1,atol=1e-10)
    facets=mesh.boundary_facets()
    mid=mesh.p[:,mesh.facets[:,facets]].mean(1)
    circle_facets=facets[np.sum((mid-np.array([[1.],[.5]]))**2,axis=0)<.2**2]
    circle=np.zeros(nu,dtype=bool);circle[bu.get_dofs(facets=circle_facets).all()]=True
    representatives=np.arange(nu)
    left=np.where(np.isclose(coords[0],0,atol=1e-10))[0]
    right=np.where(np.isclose(coords[0],2,atol=1e-10))[0]
    lookup={(round(float(coords[1,i]),10),int(component[i])):i for i in left}
    for i in right:
        representatives[i]=lookup[(round(float(coords[1,i]),10),int(component[i]))]
    free=np.where(~wall&~circle&(representatives==np.arange(nu)))[0]
    columns={int(i):k for k,i in enumerate(free)}
    rows=[];cols=[];values=[]
    g=np.zeros(nu+np_)
    for i in range(nu):
        if wall[i]:
            g[i]=(coords[1,i]-.5) if component[i]==0 else 0.
        elif circle[i]:
            rows.append(i);cols.append(len(free));values.append(-(coords[1,i]-.5) if component[i]==0 else coords[0,i]-1)
        else:
            rows.append(i);cols.append(columns[int(representatives[i])]);values.append(1.)
    # Pressure periodicity and one pressure gauge, without a penalty.
    pc=bp.doflocs;prep=np.arange(np_)
    left=np.where(np.isclose(pc[0],0,atol=1e-10))[0]
    lookup={round(float(pc[1,i]),10):i for i in left}
    for i in np.where(np.isclose(pc[0],2,atol=1e-10))[0]:prep[i]=lookup[round(float(pc[1,i]),10)]
    gauge=int(left[np.argmin(pc[1,left])])
    pf=[i for i in range(np_) if prep[i]==i and i!=gauge]
    pcolumns={i:len(free)+1+k for k,i in enumerate(pf)}
    for i in range(np_):
        if prep[i]!=gauge:
            rows.append(nu+i);cols.append(pcolumns[int(prep[i])]);values.append(1.)
    T=coo_matrix((values,(rows,cols)),shape=(nu+np_,len(free)+1+len(pf))).tocsr()
    return T,g,len(free),circle_facets


@BilinearForm
def viscous(u,v,w):return 2*ddot(sym_grad(u),sym_grad(v))


@BilinearForm
def divergence(u,q,w):return q*div(u)


@BilinearForm
def convection(u,v,w):return dot(mul(grad(u),w.velocity),v)


@BilinearForm
def convection_jacobian(u,v,w):
    return dot(mul(grad(u),w.velocity),v)+dot(mul(w.velocity.grad,u),v)


def solve(rep=1.,spacing=.06,circle_points=96,out=None):
    mesh=make_mesh(spacing,circle_points)
    bu=Basis(mesh,ElementVector(ElementTriP2()),intorder=5)
    bp=Basis(mesh,ElementTriP1(),intorder=5)
    T,g,omega_index,body_facets=constraints(bu,bp,mesh)
    K=asm(viscous,bu)/(64*rep);B=asm(divergence,bu,bp)
    L=bmat([[K,-B.T],[-B,None]],format='csr')
    reduced=(T.T@L@T).tocsc()
    y=spsolve(reduced,-T.T@(L@g))
    print(f'FEM Rep={rep}, mesh vertices={mesh.nvertices}, triangles={mesh.nelements}, unknowns={len(y)}, Stokes omega={y[omega_index]:.8f}',flush=True)
    history=[]
    for iteration in range(30):
        x=T@y+g;velocity=bu.interpolate(x[:bu.N])
        C=asm(convection,bu,velocity=velocity)
        nonlinear=np.r_[C@x[:bu.N],np.zeros(bp.N)]
        r=T.T@(L@x+nonlinear)
        error=float(np.linalg.norm(r))
        history.append(dict(iteration=iteration,residual=error,omega=float(y[omega_index])))
        print(f'FEM Newton {iteration}: residual={error:.4g}, omega={y[omega_index]:.8f}',flush=True)
        if error<1e-10:break
        J=asm(convection_jacobian,bu,velocity=velocity)
        jac=L+bmat([[J,None],[None,coo_matrix((bp.N,bp.N))]],format='csr')
        dy=spsolve((T.T@jac@T).tocsc(),-r)
        accepted=False
        for damp in [1.,.5,.25,.125,.0625]:
            yn=y+damp*dy;xn=T@yn+g
            Cn=asm(convection,bu,velocity=bu.interpolate(xn[:bu.N]))
            rn=T.T@(L@xn+np.r_[Cn@xn[:bu.N],np.zeros(bp.N)])
            if np.linalg.norm(rn)<error:
                y=yn;accepted=True;break
        if not accepted:raise RuntimeError('FEM Newton line search failed')
    else:raise RuntimeError('FEM Newton did not finish')
    x=T@y+g
    result=dict(particle_re=rep,omega=float(y[omega_index]),mesh_vertices=int(mesh.nvertices),
                mesh_triangles=int(mesh.nelements),unknowns=int(len(y)),residual=history[-1]['residual'],
                spacing=spacing,circle_points=circle_points,method='P2/P1 Taylor-Hood, symmetric viscous stress, periodic constraints, free-rotation virtual-work equation')
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'fem_solution.npz',vertices=mesh.p,triangles=mesh.t,
                            velocity=x[:bu.N],pressure=x[bu.N:],velocity_doflocs=bu.doflocs,pressure_doflocs=bp.doflocs)
        (out/'fem_metrics.json').write_text(json.dumps(result,indent=2))
        import pandas as pd
        pd.DataFrame(history).to_csv(out/'fem_newton.csv',index=False)
    return result,bu,bp,x[:bu.N],x[bu.N:]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--rep',type=float,default=1.);p.add_argument('--spacing',type=float,default=.06)
    p.add_argument('--circle-points',type=int,default=96);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result,*_=solve(a.rep,a.spacing,a.circle_points,a.out)
    print(json.dumps(result,indent=2))
