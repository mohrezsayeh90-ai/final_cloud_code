"""Steady 2D resolved-cylinder PINN for CO01 equilibrium spin.

No reference angular velocity enters the network, loss, initialization, training,
checkpoint selection or stopping criteria. Published values are read afterwards.
The symmetry-plane position is prescribed; migration is NOT validated here.
Optional FEM anchors are self-generated simulation data, never article targets.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from parcel_pinn.model import MLP
from parcel_pinn.physics import derivative
from parcel_pinn.training import seed_all


@dataclass
class Settings:
    particle_re: float = 1.
    initial_omega: float = -.3
    seed: int = 11
    width: int = 40
    depth: int = 3
    adam_steps: int = 2000
    lbfgs_steps: int = 400
    batch: int = 192
    fixed_points: int = 1024
    torque_points: int = 96
    lr: float = .002
    threads: int = 2
    radius: float = .125
    length: float = 2.
    height: float = 1.
    center_y: float = .5
    symmetry_translation: float = .5
    comparison_relative_tolerance: float = .10
    residual_tolerance: float = .01
    torque_tolerance: float = .002
    formulation: str = 'velocity'
    teacher: str | None = None
    teacher_samples: int = 1024
    data_weight: float = 20.
    feature_set: str = 'cartesian'

    @property
    def nu(self):
        return 1/(64*self.particle_re)


def strain_traction(field, points, normals):
    u, v = field[:, 0], field[:, 1]
    gu, gv = derivative(u, points), derivative(v, points)
    n1, n2 = normals.unbind(1)
    sx = 2*gu[:, 0]*n1+(gu[:, 1]+gv[:, 0])*n2
    sy = (gu[:, 1]+gv[:, 0])*n1+2*gv[:, 1]*n2
    return torch.stack([sx, sy], 1)


def steady_operator(f, xy, nu):
    gu, gv, gp = [derivative(f[:, i], xy) for i in range(3)]
    lu = derivative(gu[:, 0], xy)[:, 0]+derivative(gu[:, 1], xy)[:, 1]
    lv = derivative(gv[:, 0], xy)[:, 0]+derivative(gv[:, 1], xy)[:, 1]
    rx = f[:, 0]*gu[:, 0]+f[:, 1]*gu[:, 1]+gp[:, 0]-nu*lu
    ry = f[:, 0]*gv[:, 0]+f[:, 1]*gv[:, 1]+gp[:, 1]-nu*lv
    return torch.stack([rx, ry], 1), gu[:, 0]+gv[:, 1]


class CylinderPINN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.net = MLP(9 if settings.feature_set=='multipole' else 7, 2, settings.width, settings.depth).double()
        self.omega = nn.Parameter(torch.tensor(settings.initial_omega, dtype=torch.float64))
        self.stream_constant = nn.Parameter(torch.tensor(.25, dtype=torch.float64))

    def geometry(self, xy):
        dx, dy = xy[:, 0]-1., xy[:, 1]-.5
        r = (dx*dx+dy*dy).clamp_min(1e-18).sqrt()
        s = ((r-self.settings.radius)/(.45-self.settings.radius)).clamp(max=1.)
        # chi(R)=1, chi'(R)=0, chi''(R) nonzero; derivatives through
        # order 3 vanish at the outer edge, avoiding a fictitious stress sheet.
        chi = (1-s)**4*(1+4*s)
        return dx, dy, r, chi

    def features(self, xy):
        dx, dy, r, chi = self.geometry(xy)
        x = xy[:, 0]
        base=[(math.pi*x).sin(), (math.pi*x).cos(),
                            (2*math.pi*x).sin(), (2*math.pi*x).cos(),
                            2*dy]
        if self.settings.feature_set=='multipole':
            s=((r-self.settings.radius)/(.9-self.settings.radius)).clamp(max=1.)
            b=(1-s)**4;R=self.settings.radius
            base += [b*R*dx/r**2,b*R*dy/r**2,
                     b*R**2*(dx*dx-dy*dy)/r**4,b*R**2*(2*dx*dy)/r**4]
        else:base += [chi*dx,chi*dy]
        return torch.stack(base,1)

    def raw(self, xy):
        reflected = torch.stack([2-xy[:, 0], 1-xy[:, 1]], 1)
        return .5*(self.net(self.features(xy))+self.net(self.features(reflected)))

    def stream_pressure(self, xy):
        dx, dy, r, chi = self.geometry(xy)
        base = .5*dy*dy
        solid = .02*self.stream_constant-.5*self.omega*(r*r-self.settings.radius**2)
        raw = self.raw(xy)
        bubble = 16*xy[:, 1]**2*(1-xy[:, 1])**2*(1-chi)
        psi = (1-chi)*base+chi*solid+.03*bubble*raw[:, 0]
        anchor = xy.new_tensor([[0., .5]])
        p = .1*(raw[:, 1]-self.raw(anchor)[0, 1])
        return psi, p

    def forward(self, xy):
        psi, p = self.stream_pressure(xy)
        grad = derivative(psi, xy)
        return torch.stack([grad[:, 1], -grad[:, 0], p], 1)

    def residual(self, xy):
        xy = xy.detach().requires_grad_(True)
        f = self(xy)
        return steady_operator(f,xy,self.settings.nu)

    def surface(self, count, offset=0.):
        theta = (torch.arange(count, dtype=torch.float64)+.37)*2*math.pi/count
        normals = torch.stack([theta.cos(), theta.sin()], 1)
        points = normals*(self.settings.radius+offset)+normals.new_tensor([1., .5])
        return points.requires_grad_(True), normals

    def torque(self, count=None, offset=0.):
        xy, n = self.surface(count or self.settings.torque_points, offset)
        stress = strain_traction(self(xy), xy, n)
        # T/(4*pi*mu*R^2): factor 1/2 from the 2*pi quadrature.
        return .5*(-n[:, 1]*stress[:, 0]+n[:, 0]*stress[:, 1]).mean()

    def loss(self, xy):
        r, div = self.residual(xy)
        torque = self.torque()
        ns = r.square().mean()
        return ns+5*torque.square(), dict(momentum=ns, torque=torque.square())


class VelocityPINN(CylinderPINN):
    """Mixed velocity/pressure PINN: second-order spatial AD and hard no-slip.

    Continuity is trained explicitly. Boundary normal velocity and torque are
    never prescribed from literature data. Centrosymmetry gives zero net force.
    """
    def __init__(self,settings):
        nn.Module.__init__(self)
        self.settings=settings
        self.net=MLP(9 if settings.feature_set=='multipole' else 7,3,settings.width,settings.depth).double()
        self.omega=nn.Parameter(torch.tensor(settings.initial_omega,dtype=torch.float64))

    def geometry(self,xy):
        dx,dy=xy[:,0]-1.,xy[:,1]-.5
        r=(dx*dx+dy*dy).clamp_min(1e-18).sqrt()
        s=((r-self.settings.radius)/(.45-self.settings.radius)).clamp(max=1.)
        return dx,dy,r,(1-s)**4

    def raw(self,xy):
        reflected=torch.stack([2-xy[:,0],1-xy[:,1]],1)
        a,b=self.net(self.features(xy)),self.net(self.features(reflected))
        return torch.cat([.5*(a[:,:2]-b[:,:2]),.5*(a[:,2:]+b[:,2:])],1)

    def forward(self,xy):
        dx,dy,r,chi=self.geometry(xy)
        raw=self.raw(xy)
        envelope=4*xy[:,1]*(1-xy[:,1])*(1-chi)
        u=(1-chi)*dy+chi*(-self.omega*dy)+.3*envelope*raw[:,0]
        v=chi*(self.omega*dx)+.3*envelope*raw[:,1]
        anchor=xy.new_tensor([[0.,.5]])
        p=.1*(raw[:,2]-self.raw(anchor)[0,2])
        return torch.stack([u,v,p],1)

    def loss(self,xy):
        r,div=self.residual(xy)
        torque=self.torque()
        ns=r.square().mean();continuity=div.square().mean()
        return ns+3*continuity+5*torque.square(),dict(momentum=ns,continuity=continuity,torque=torque.square())


def sample_points(count, generator, near_fraction=.5):
    near = int(count*near_fraction)
    uniform = []
    while sum(len(a) for a in uniform)<count-near:
        xy = torch.rand(count, 2, generator=generator, dtype=torch.float64)*torch.tensor([2., 1.])
        distance = (xy[:, 0]-1.)**2+(xy[:, 1]-.5)**2
        uniform.append(xy[distance>.125**2])
    xy = torch.cat(uniform)[:count-near]
    theta = torch.rand(near, generator=generator, dtype=torch.float64)*2*math.pi
    radius = .125+(.45-.125)*torch.rand(near, generator=generator, dtype=torch.float64).square()+1e-7
    annulus = torch.stack([1+radius*theta.cos(), .5+radius*theta.sin()], 1)
    return torch.cat([xy, annulus])


def train(settings, out):
    seed_all(settings.seed, settings.threads)
    model = CylinderPINN(settings) if settings.formulation=='streamfunction' else VelocityPINN(settings)
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    (out/'settings.json').write_text(json.dumps(asdict(settings), indent=2))
    generator = torch.Generator().manual_seed(settings.seed+100)
    audit = sample_points(512, torch.Generator().manual_seed(99331))
    fixed = sample_points(settings.fixed_points, torch.Generator().manual_seed(54907))
    teacher_points=teacher_velocity=None
    if settings.teacher:
        from skfem import MeshTri,Basis,ElementVector,ElementTriP2
        with np.load(settings.teacher) as z:
            mesh=MeshTri(z['vertices'],z['triangles'])
            basis=Basis(mesh,ElementVector(ElementTriP2()))
            if not np.allclose(basis.doflocs,z['velocity_doflocs']):
                raise ValueError('FEM basis ordering mismatch')
            teacher_points=sample_points(settings.teacher_samples,torch.Generator().manual_seed(78041))
            uv=basis.interpolator(z['velocity'])(teacher_points.numpy().T).T
            teacher_velocity=torch.tensor(uv,dtype=torch.float64)
        pd.DataFrame(dict(x=teacher_points[:,0],y=teacher_points[:,1],u=uv[:,0],v=uv[:,1])).to_csv(out/'internal_fem_anchors.csv',index=False)
    def objective(xy):
        loss,terms=model.loss(xy)
        if teacher_points is not None:
            uv=model(teacher_points.requires_grad_(True))[:,:2]
            anchor_loss=(uv-teacher_velocity).square().mean()
            loss=loss+settings.data_weight*anchor_loss
            terms['internal_FEM_anchors']=anchor_loss
        return loss,terms
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.2**(1/max(settings.adam_steps, 1)))
    rows = []; best = float('inf'); best_state = None
    started = time.monotonic()
    def record(step, stage):
        nonlocal best,best_state
        loss, terms = objective(audit)
        value = float(loss.detach())
        if not math.isfinite(value):raise FloatingPointError('nonfinite audit')
        if value<best:
            best=value;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        rows.append(dict(step=step,stage=stage,loss=value,omega=float(model.omega.detach()),
                         **{k:float(v.detach()) for k,v in terms.items()}))
        if step==0 or step%max(100,settings.adam_steps//10)==0:
            print(f'Rep={settings.particle_re:g}, step={step}, residual loss={value:.6g}, omega={float(model.omega.detach()):.6f}',flush=True)
    reason='budget_exhausted'
    try:
        record(0,'initial')
        for step in range(1,settings.adam_steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss,_=objective(sample_points(settings.batch,generator))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite Adam loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();scheduler.step()
            if step==1 or step%max(20,settings.adam_steps//60)==0 or step==settings.adam_steps:
                record(step,'Adam')
        model.load_state_dict(best_state)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=1,history_size=40,line_search_fn='strong_wolfe')
        for step in range(1,settings.lbfgs_steps+1):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss,_=objective(fixed)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite L-BFGS loss')
                loss.backward()
                return loss
            optimizer.step(closure)
            if step%5==0 or step==settings.lbfgs_steps:
                record(settings.adam_steps+step,'L-BFGS')
    except (RuntimeError,FloatingPointError) as exc:
        reason=f'{type(exc).__name__}: {exc}'
    if best_state is not None:model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(out/'training.csv',index=False)
    torch.save(model.state_dict(),out/'checkpoint.pt')
    (out/'training_status.json').write_text(json.dumps(dict(termination=reason,seconds=time.monotonic()-started,best_loss=best,
        mode='simulation-assisted PINN' if settings.teacher else 'physics-only PINN',
        teacher_sha256=hashlib.sha256(Path(settings.teacher).read_bytes()).hexdigest() if settings.teacher else None,
        literature_values_used_in_training=False),indent=2))
    return model


def assess(model):
    generator=torch.Generator().manual_seed(85812)
    values=[];divs=[]
    for chunk in torch.split(sample_points(2048,generator),256):
        residual,div=model.residual(chunk)
        values.append(residual.detach().numpy());divs.append(div.detach().numpy())
    c=model.settings
    xy,n=model.surface(512)
    f=model(xy)
    boundary=torch.stack([-model.omega*(xy[:,1]-.5),model.omega*(xy[:,0]-1)],1)
    surface_error=float((f[:,:2]-boundary).abs().max().detach())
    wall=torch.stack([torch.linspace(0,2,101,dtype=torch.float64).repeat(2),
                      torch.cat([torch.zeros(101,dtype=torch.float64),torch.ones(101,dtype=torch.float64)])],1).requires_grad_(True)
    fw=model(wall)
    wall_velocity=torch.stack([wall[:,1]-.5,torch.zeros_like(wall[:,1])],1)
    return dict(omega=float(model.omega.detach()),
                momentum_rms=float(np.sqrt(np.mean(np.concatenate(values)**2))),
                continuity_rms=float(np.sqrt(np.mean(np.concatenate(divs)**2))),
                torque_normalized=float(model.torque(1024).detach()),
                torque_normalized_256=float(model.torque(256).detach()),
                torque_offset_1e5=float(model.torque(512,1e-5).detach()),
                cylinder_boundary_max_error=surface_error,
                wall_boundary_max_error=float((fw[:,:2]-wall_velocity).abs().max().detach()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rep',type=float,choices=[1.,2.],default=1.)
    p.add_argument('--adam',type=int,default=2000)
    p.add_argument('--lbfgs',type=int,default=400)
    p.add_argument('--batch',type=int,default=192)
    p.add_argument('--fixed-points',type=int,default=1024)
    p.add_argument('--width',type=int,default=40)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--initial-omega',type=float,default=-.3)
    p.add_argument('--formulation',choices=['velocity','streamfunction'],default='velocity')
    p.add_argument('--teacher',type=Path,help='Independent FEM velocity solution for optional internal anchors; no article values used')
    p.add_argument('--feature-set',choices=['cartesian','multipole'],default='multipole')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pack',type=Path,default=Path('reference_inputs/Validation_Pack'))
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):p.error('output directory must be empty')
    cfg=Settings(particle_re=args.rep,adam_steps=args.adam,lbfgs_steps=args.lbfgs,batch=args.batch,
                 fixed_points=args.fixed_points,width=args.width,seed=args.seed,initial_omega=args.initial_omega,
                 formulation=args.formulation,teacher=str(args.teacher) if args.teacher else None,feature_set=args.feature_set)
    model=train(cfg,args.out)
    result=assess(model)
    # References are accessed only after optimization and checkpoint selection.
    reference_file=args.pack/'data/couette_equilibrium.csv'
    table=pd.read_csv(reference_file)
    row=table[np.isclose(table.Re,args.rep)].iloc[0]
    reference=float(row.omega_magnitude_H_over_dU)
    result.update(reference_omega_magnitude=reference,predicted_omega_magnitude=abs(result['omega']),
                  relative_error=abs(abs(result['omega'])-reference)/reference,
                  reference_sha256=hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                  scope='Conditional equilibrium-spin validation at prescribed symmetry-plane center. No migration trajectory validation; no validation of the original point-parcel closure.')
    result['comparison_within_tolerance']=result['relative_error']<=cfg.comparison_relative_tolerance
    result['physics_criteria_met']=(result['momentum_rms']<cfg.residual_tolerance and result['continuity_rms']<cfg.residual_tolerance and
                                   abs(result['torque_normalized'])<cfg.torque_tolerance and
                                   result['cylinder_boundary_max_error']<1e-10 and result['wall_boundary_max_error']<1e-10)
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
