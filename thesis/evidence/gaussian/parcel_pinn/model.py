import math
import numpy as np
import torch
from torch import nn
from .physics import derivative, flow_terms, gaussian_raw, grid, interpolate, soft_contact, spread, weights


class MLP(nn.Module):
    def __init__(self, inputs, outputs, width, depth):
        super().__init__()
        layers = [nn.Linear(inputs, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, outputs)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.net[-1].weight *= .02

    def forward(self, x):
        return self.net(x)


class CoupledPINN(nn.Module):
    def __init__(self, cfg, case):
        super().__init__()
        self.cfg, self.case = cfg, case
        self.physical = cfg.physical(case)
        self.D = cfg.dimension
        self.fluid = MLP(2*(self.D-1)*cfg.fourier_modes+2, self.D+1, cfg.hidden, cfg.depth)
        self.parcels = MLP(1, 2*self.D*cfg.nparcels, cfg.hidden, cfg.depth)
        rng = np.random.default_rng(cfg.seed)
        q0 = rng.uniform(np.array([.2, .25, .18])[:self.D], np.array([1.3, .75, .82])[:self.D], (cfg.nparcels, self.D))
        q0[:, 0] *= cfg.lengths[0]/3
        if self.D == 3:
            q0[:, 2] *= cfg.lengths[2]
        self.register_buffer('q0', torch.tensor(q0, dtype=torch.float64))
        self.register_buffer('v0', torch.zeros(cfg.nparcels, self.D, dtype=torch.float64))
        xyz, self.dv = grid(cfg.lengths, cfg.quadrature)
        self.register_buffer('quad', xyz)
        self.double()

    def features(self, xt):
        x, y, t = xt[:, 0], xt[:, 1], xt[:, -1]
        f = [2*y-1]
        for d in range(self.D):
            if d == 1:continue
            for mode in range(1,self.cfg.fourier_modes+1):
                angle = 2*math.pi*mode*xt[:,d]/self.cfg.lengths[d]
                f += [angle.sin(),angle.cos()]
        return torch.stack(f+[2*t/self.cfg.final_time-1], 1)

    def field(self, xt):
        r = self.fluid(self.features(xt))
        y, t = xt[:, 1:2], xt[:, -1:]
        initial = torch.cat([6*y*(1-y), torch.zeros_like(y).expand(-1, self.D-1)], 1)
        u = initial+(t/self.cfg.final_time)*y*(1-y)*r[:, :self.D]
        anchor = torch.cat([torch.zeros_like(xt[:, :self.D]), t], 1)
        # Pressure is elliptic, not an independently prescribed initial state.
        # Localized initial drag generally requires a nonzero pressure field.
        p = r[:, self.D:]-self.fluid(self.features(anchor))[:, self.D:]
        return torch.cat([u, p], 1)

    def particle_state(self, t):
        raw = self.parcels(2*t/self.cfg.final_time-1).reshape(-1, self.cfg.nparcels, 2*self.D)
        if self.cfg.relaxation_basis:
            # Physics-scaled envelopes, not reference targets: NN_q and NN_v
            # remain independent trainable functions of time in the coupled system.
            tau=self.case['stokes']
            b=-torch.expm1(-t/tau)
            a=t-tau*b
        else:
            a,b=t*t/self.cfg.final_time,t
        q = self.q0+t[:, :, None]*self.v0+a[:, :, None]*raw[..., :self.D]
        v = self.v0+b[:, :, None]*raw[..., self.D:]
        return q, v

    def coupling(self, t):
        q, v = self.particle_state(t)
        xyz = self.quad[None].expand(len(t), -1, -1)
        xt = torch.cat([xyz, t[:, None].expand(-1, len(self.quad), -1)], -1)
        u = self.field(xt.reshape(-1, self.D+1))[:, :self.D].reshape(len(t), -1, self.D)
        W, Z = weights(xyz, q, self.cfg.width, self.cfg.lengths, self.dv)
        U = interpolate(W, u, self.dv)
        drag = (U-v)*self.physical['represented_mass']/self.case['stokes']
        return q, v, U, W, Z, drag

    def residuals(self, t, xyz):
        t = t.detach().requires_grad_(True)
        q, v, U, W, Z, force = self.coupling(t)
        T, S = xyz.shape[:2]
        xt = torch.cat([xyz, t[:, None].expand(-1, S, -1)], -1).reshape(-1, self.D+1)
        xt = xt.detach().requires_grad_(True)
        ns, div, _ = flow_terms(self.field(xt), xt, 1/self.cfg.reynolds)
        source_weights = gaussian_raw(xyz, q, self.cfg.width, self.cfg.lengths)/Z
        reaction = spread(source_weights, force)
        drive = xt.new_tensor([12/self.cfg.reynolds]+[0.]*(self.D-1))
        momentum = ns.reshape(T, S, self.D)-reaction-drive
        dq = torch.stack([derivative(q[:, i, d], t)[:, 0]
                          for i in range(self.cfg.nparcels) for d in range(self.D)], 1).reshape_as(q)
        dv = torch.stack([derivative(v[:, i, d], t)[:, 0]
                          for i in range(self.cfg.nparcels) for d in range(self.D)], 1).reshape_as(v)
        acc = (U-v)/self.case['stokes']+t.new_tensor(self.cfg.gravity)
        if self.cfg.collision == 'soft_sphere':
            acc = acc+soft_contact(q, v, self.physical['diameter'], self.physical['physical_mass'],
                                  self.cfg.lengths, self.cfg.stiffness, self.cfg.damping)
        confinement = torch.relu(-q[..., 1])+torch.relu(q[..., 1]-1)
        return dict(momentum=momentum, continuity=div, position=dq-v,
                    particle=dv-acc, confinement=confinement)

    def loss(self, t, xyz):
        residuals = self.residuals(t, xyz)
        scales = dict(momentum=1., continuity=2., position=5., particle=1., confinement=100.)
        terms = {k: (r*r).mean() for k, r in residuals.items()}
        return sum(scales[k]*v for k, v in terms.items()), terms
