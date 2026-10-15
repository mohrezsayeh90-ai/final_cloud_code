from pathlib import Path
import numpy as np
import pandas as pd
import torch
from parcel_pinn.physics import derivative


def evaluate(model, folder):
    c, geo = model.cfg, model.geometry
    m = model.physical['represented_mass']; diameter = model.physical['diameter']
    times = np.linspace(0, c.final_time, c.frames)
    points, normals, measures = geo.boundary()
    normal = torch.tensor(normals, dtype=torch.float64)
    ds = torch.tensor(measures, dtype=torch.float64)
    particles, budgets, fields, concentrations = [], [], [], []
    for value in times:
        t = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
        q, v, U, W, Z, force = model.coupling(t)
        xt = torch.cat([model.quad, t.expand(len(model.quad), 1)], 1)
        uvp = model.field(xt); u = uvp[:, :2]
        qn, vn = q[0].detach().numpy(), v[0].detach().numpy()
        if not all(np.isfinite(a).all() for a in [qn, vn, uvp.detach().numpy()]):
            raise FloatingPointError('Nonfinite PINN prediction; no synthetic replacement field.')
        momentum = u.sum(0)*model.dv+m*v.sum((0, 1))
        dp = torch.stack([derivative(momentum[k], t)[0, 0] for k in range(2)])
        energy = .5*u.square().sum()*model.dv+.5*m*v.square().sum()
        de = derivative(energy, t)[0, 0]
        # Full curved wall traction. Periodic end-face tractions cancel.
        wall = torch.tensor(np.column_stack([points, np.full(len(points), value)]),
                            dtype=torch.float64, requires_grad=True)
        fw = model.field(wall)
        du, dv = derivative(fw[:, 0], wall)[:, :2], derivative(fw[:, 1], wall)[:, :2]
        tx = -fw[:, 2]*normal[:, 0]+(2*du[:, 0]*normal[:, 0]+(du[:, 1]+dv[:, 0])*normal[:, 1])/c.reynolds
        ty = -fw[:, 2]*normal[:, 1]+((du[:, 1]+dv[:, 0])*normal[:, 0]+2*dv[:, 1]*normal[:, 1])/c.reynolds
        traction = torch.stack([(tx*ds).sum(), (ty*ds).sum()])
        external = t.new_tensor([12/c.reynolds*model.fluid_volume_quadrature, 0.])+m*c.nparcels*t.new_tensor(c.gravity)
        momentum_error = dp-traction-external
        slip = U-v
        diss_drag = m/model.case['stokes']*slip.square().sum()
        independent = xt.detach().requires_grad_(True)
        velocity = model.field(independent)[:, :2]
        grad = [derivative(velocity[:, k], independent)[:, :2] for k in range(2)]
        diss_visc = sum(g.square().sum() for g in grad)*model.dv/c.reynolds
        power = 12/c.reynolds*u[:, 0].sum()*model.dv+m*(v*t.new_tensor(c.gravity)).sum()
        alpha = (W[0]*m/c.density_ratio).sum(1)
        inside = ~geo.fluid(qn)
        outside_channel = (qn[:, 1] < 0) | (qn[:, 1] > 1)
        obstacle_inside = inside & ~outside_channel
        rep = c.reynolds*diameter*torch.linalg.vector_norm(slip, dim=-1)
        budgets.append(dict(time=value, energy=float(energy.detach()),
            momentum_x=float(momentum[0].detach()), momentum_y=float(momentum[1].detach()),
            momentum_balance_residual=float(torch.linalg.vector_norm(momentum_error).detach()),
            energy_balance_residual=float((de+diss_drag+diss_visc-power).detach()),
            drag_dissipation=float(diss_drag.detach()), viscous_dissipation=float(diss_visc.detach()),
            kernel_error=float((W.sum(1)*model.dv-1).abs().max().detach()),
            max_slip_re=float(rep.max().detach()), max_volume_fraction=float(alpha.max().detach()),
            wall_outside_count=int(outside_channel.sum()), obstacle_inside_count=int(obstacle_inside.sum()),
            boundary_velocity_max=float(fw[:, :2].abs().max().detach()),
            force_exchange_error=float((-(W*model.dv).sum(1)[:, :, None]*force+force).abs().max().detach())))
        sampled = U[0].detach().numpy()
        for i in range(c.nparcels):
            particles.append(dict(time=value, particle=i, x=qn[i, 0] % c.lengths[0], y=qn[i, 1], z=0.,
                x_unwrapped=qn[i, 0], z_unwrapped=0., vx=vn[i, 0], vy=vn[i, 1], vz=0.,
                sampled_ux=sampled[i, 0], sampled_uy=sampled[i, 1], stokes=model.case['stokes'],
                loading=model.physical['loading'], multiplicity=model.physical['multiplicity'], diameter=diameter,
                inside_solid=bool(obstacle_inside[i]), outside_channel=bool(outside_channel[i])))
        fields.append(uvp.detach().numpy()); concentrations.append(alpha.detach().numpy())
    generator = torch.Generator().manual_seed(991)
    tt = torch.rand(9, 1, generator=generator, dtype=torch.float64)*c.final_time
    tt[0]=0.; tt[1]=.01*c.final_time; tt[2]=.05*c.final_time; tt[-1]=c.final_time
    xyz = geo.sample(9*79, generator).reshape(9, 79, 2)
    residuals = {k: float(v.square().mean().sqrt().detach()) for k, v in model.residuals(tt, xyz).items()}
    particle_table, budget = pd.DataFrame(particles), pd.DataFrame(budgets)
    particle_table.to_csv(folder/'particles.csv', index=False); budget.to_csv(folder/'diagnostics.csv', index=False)
    np.savez_compressed(folder/'fields.npz', times=times, xyz=model.quad.detach().numpy(),
                        velocity_pressure=np.array(fields), volume_fraction=np.array(concentrations),
                        cell_area=model.dv, fluid_volume_quadrature=model.fluid_volume_quadrature,
                        fluid_volume_exact=model.fluid_volume_exact)
    return particle_table, budget, residuals
