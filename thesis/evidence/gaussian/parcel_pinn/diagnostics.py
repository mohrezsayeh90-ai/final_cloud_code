import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .physics import derivative, flow_terms


def evaluate(model, out):
    c = model.cfg
    times = np.linspace(0, c.final_time, c.frames)
    particle_rows, budgets, states = [], [], []
    m = model.physical['represented_mass']
    diameter = model.physical['diameter']
    for tvalue in times:
        t = torch.tensor([[tvalue]], dtype=torch.float64, requires_grad=True)
        q, v, U, W, Z, force = model.coupling(t)
        xyz = model.quad
        xt = torch.cat([xyz, t.expand(len(xyz), 1)], 1)
        D = c.dimension
        u = model.field(xt)[:, :D]
        qn, vn, un = q[0].detach().numpy(), v[0].detach().numpy(), u.detach().numpy()
        if not all(np.isfinite(a).all() for a in (qn, vn, un)):
            raise FloatingPointError('nonfinite predictions: no synthetic replacement generated')
        momentum = u.sum(0)*model.dv+m*v.sum((0, 1))
        dp = torch.stack([derivative(momentum[k], t)[0, 0] for k in range(D)])
        energy = .5*u.square().sum()*model.dv+.5*m*v.square().sum()
        de = derivative(energy, t)[0, 0]
        # Integrate viscous wall traction; x/z surfaces cancel by periodicity.
        nx = c.quadrature[0]
        gx = (torch.arange(nx, dtype=torch.float64)+.5)*c.lengths[0]/nx
        if D == 3:
            nz = c.quadrature[2]
            gz = (torch.arange(nz, dtype=torch.float64)+.5)*c.lengths[2]/nz
            xx, zz = torch.meshgrid(gx, gz, indexing='ij')
        else:
            xx, zz = gx, None
        area = c.lengths[0]*(c.lengths[2] if D == 3 else 1.)
        def wall_points(yy):
            coords = [xx.ravel(), torch.full_like(xx.ravel(), yy)]
            if D == 3:
                coords += [zz.ravel()]
            return torch.stack(coords+[torch.full_like(xx.ravel(), tvalue)], 1)
        traction = torch.zeros(D, dtype=torch.float64)
        for yy, sign in [(0., -1.), (1., 1.)]:
            wall = wall_points(yy).requires_grad_(True)
            fw = model.field(wall)
            traction = traction+sign*torch.stack([derivative(fw[:, k], wall)[:, 1].mean() for k in range(D)])/c.reynolds*area
        external = t.new_tensor([12/c.reynolds*math.prod(c.lengths)]+[0.]*(D-1))+m*c.nparcels*t.new_tensor(c.gravity)
        # Pressure traction at the two y walls is required in the y momentum budget.
        pressures = []
        for yy in (0., 1.):
            pts = wall_points(yy)
            pressures.append(model.field(pts)[:, -1].mean())
        pressure_force = t.new_zeros(D)
        pressure_force[1] = (pressures[0]-pressures[1])*area
        budget_error = dp-traction-external-pressure_force
        slip = U-v
        diss_drag = m/model.case['stokes']*slip.square().sum()
        independent_xt = xt.detach().requires_grad_(True)
        uvp = model.field(independent_xt)
        gradients = [derivative(uvp[:, k], independent_xt) for k in range(D)]
        diss_visc = sum(g[:, :D].square().sum() for g in gradients)*model.dv/c.reynolds
        external_power = (12/c.reynolds*u[:, 0].sum()*model.dv+m*(v*t.new_tensor(c.gravity)).sum())
        energy_error = de+diss_drag+diss_visc-external_power
        alpha = (W[0]*m/c.density_ratio).sum(1)
        closure = (W.sum(1)*model.dv-1).abs().max()
        rep = c.reynolds*diameter*torch.linalg.vector_norm(slip, dim=-1)
        budgets.append(dict(time=tvalue, energy=float(energy.detach()),
                            drag_dissipation=float(diss_drag.detach()),
                            energy_balance_residual=float(energy_error.detach()) if c.collision == 'none' else np.nan,
                            momentum_balance_residual=float(torch.linalg.vector_norm(budget_error).detach()) if c.collision == 'none' else np.nan,
                            momentum_x=float(momentum[0].detach()), momentum_y=float(momentum[1].detach()),
                            kernel_error=float(closure.detach()), max_slip_re=float(rep.max().detach()),
                            max_volume_fraction=float(alpha.max().detach()),
                            wall_outside_count=int(((q[..., 1]<0)|(q[..., 1]>1)).sum()),
                            viscous_dissipation=float(diss_visc.detach())))
        for i in range(c.nparcels):
            wrapped = qn[i].copy()
            for axis in range(D):
                if axis != 1:
                    wrapped[axis] %= c.lengths[axis]
            particle_rows.append(dict(time=tvalue, particle=i, x=wrapped[0], y=wrapped[1], z=wrapped[2] if D == 3 else 0.,
                                      x_unwrapped=qn[i, 0], z_unwrapped=qn[i, 2] if D == 3 else 0.,
                                      vx=vn[i, 0], vy=vn[i, 1], vz=vn[i, 2] if D == 3 else 0.,
                                      sampled_ux=float(U[0, i, 0].detach()),
                                      stokes=model.case['stokes'], loading=model.physical['loading'],
                                      multiplicity=model.physical['multiplicity'], diameter=diameter))
        states.append((un, alpha.detach().numpy()))
    # Independent held-out residual locations, generated separately from training/audit.
    gen = torch.Generator().manual_seed(991)
    tt = torch.rand(9, 1, generator=gen, dtype=torch.float64)*c.final_time
    tt[0]=0.;tt[1]=c.final_time*.01;tt[2]=c.final_time*.05;tt[-1]=c.final_time
    xx = torch.rand(9, 79, c.dimension, generator=gen, dtype=torch.float64)*torch.tensor(c.lengths)
    residuals = model.residuals(tt, xx)
    residual_summary = {k: float(v.square().mean().sqrt().detach()) for k, v in residuals.items()}
    particles, budget = pd.DataFrame(particle_rows), pd.DataFrame(budgets)
    particles.to_csv(Path(out)/'particles.csv', index=False)
    budget.to_csv(Path(out)/'diagnostics.csv', index=False)
    np.savez_compressed(Path(out)/'fields.npz', times=times, xyz=model.quad.detach().numpy(),
                        velocity=np.array([s[0] for s in states]), volume_fraction=np.array([s[1] for s in states]))
    return particles, budget, residual_summary
