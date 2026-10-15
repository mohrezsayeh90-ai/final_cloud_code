from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import torch
from parcel_pinn.model import CoupledPINN
from parcel_pinn.physics import derivative
from parcel_pinn.training import fit


@dataclass
class GeometrySettings:
    radius: float = .25
    center_fraction: float = .4

    def validate(self, length):
        xc = length*self.center_fraction
        if not np.isfinite([self.radius, self.center_fraction, length]).all():
            raise ValueError('Geometry parameters must be finite.')
        if not 0 < self.center_fraction < 1 or not 0 <= self.radius < .5:
            raise ValueError('Use 0 <= radius/H < .5 and 0 < center_fraction < 1.')
        if self.radius >= min(xc, length-xc, .45*length):
            raise ValueError('Semicircles must be clear of the periodic end faces.')
        return self

    def save(self, path): Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path): return cls(**json.loads(Path(path).read_text()))


class Geometry:
    def __init__(self, length, settings):
        settings.validate(length)
        self.length, self.radius = length, settings.radius
        self.xc = length*settings.center_fraction
        self.support = min(2*self.radius, .45*length)

    def dx(self, x):
        if torch.is_tensor(x): return x-self.xc-self.length*torch.floor((x-self.xc)/self.length+.5)
        return x-self.xc-self.length*np.floor((x-self.xc)/self.length+.5)

    def fluid(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        dx = self.dx(x)
        return (y >= 0) & (y <= 1) & (dx*dx+y*y >= self.radius**2) & (dx*dx+(1-y)**2 >= self.radius**2)

    def bounds(self, x):
        lo = np.sqrt(np.maximum(self.radius**2-self.dx(np.asarray(x))**2, 0))
        return lo, 1-lo

    def factors(self, xy):
        y = xy[..., 1:2]
        if not self.radius: return torch.ones_like(y), torch.ones_like(y)
        dx = self.dx(xy[..., :1]); r = self.radius
        s = ((dx.abs()-r)/(self.support-r)).clamp(0, 1)
        # C3 compact cutoff, identically zero near the periodic seam.
        gate = (1-s)**4/((1-s)**4+s**4)
        d1, d2 = dx*dx+y*y-r*r, dx*dx+(1-y)**2-r*r
        outside = dx.abs() >= self.support
        b = torch.where(outside, torch.ones_like(d1), d1/(d1+r*r*gate))
        t = torch.where(outside, torch.ones_like(d2), d2/(d2+r*r*gate))
        return b, t

    def initial_velocity(self, xy):
        bottom, top = self.factors(xy)
        y = xy[:, 1:2]
        a, b = y*bottom, (1-y)*top
        s = a/(a+b)
        psi = 3*s*s-2*s*s*s
        dp = derivative(psi[:, 0], xy)
        return torch.stack([dp[:, 1], -dp[:, 0]], 1)

    def sample(self, n, generator, *, dtype=torch.float64, device='cpu'):
        pieces, count = [], 0
        while count < n:
            points = torch.rand(max(64, 2*(n-count)), 2, generator=generator, dtype=dtype, device=device)
            points[:, 0] *= self.length
            accepted = points[self.fluid(points)]
            pieces.append(accepted); count += len(accepted)
        return torch.cat(pieces)[:n]

    def boundary(self, order=80):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        points, normals, measures = [], [], []
        intervals = [(0., self.length)] if not self.radius else [(0., self.xc-self.radius), (self.xc+self.radius, self.length)]
        for y, normal in [(0., [0., -1.]), (1., [0., 1.])]:
            for a, b in intervals:
                x = a+(nodes+1)*(b-a)/2
                points.append(np.column_stack([x, np.full(len(x), y)]))
                normals.append(np.tile(normal, (len(x), 1))); measures.append(weights*(b-a)/2)
        if self.radius:
            theta = (nodes+1)*np.pi/2; r = self.radius
            for upper in [False, True]:
                y = 1-r*np.sin(theta) if upper else r*np.sin(theta)
                points.append(np.column_stack([self.xc+r*np.cos(theta), y]))
                normals.append(np.column_stack([-np.cos(theta), np.sin(theta) if upper else -np.sin(theta)]))
                measures.append(weights*r*np.pi/2)
        return np.concatenate(points), np.concatenate(normals), np.concatenate(measures)


class ConstrictedGaussianPINN(CoupledPINN):
    """Same fluid/parcel networks, drag, ODEs and force spreading as CoupledPINN."""
    def __init__(self, cfg, case, geometry):
        if cfg.dimension != 2 or cfg.collision != 'none':
            raise ValueError('This extension preserves the original 2D dilute collision-free model.')
        super().__init__(cfg, case)
        self.geometry_settings = geometry
        self.geometry = Geometry(cfg.lengths[0], geometry)
        valid = self.geometry.fluid(self.quad)
        self.quad = self.quad[valid]
        if len(self.quad) < 8: raise ValueError('Too few fluid quadrature points; increase quadrature.')
        self.initial_centers_resampled = 0
        bad = ~self.geometry.fluid(self.q0)
        if bad.any():
            gen = torch.Generator().manual_seed(cfg.seed+941)
            with torch.no_grad(): self.q0[bad] = self.geometry.sample(int(bad.sum()), gen)
            self.initial_centers_resampled = int(bad.sum())
        self.fluid_volume_quadrature = len(self.quad)*self.dv
        self.fluid_volume_exact = np.prod(cfg.lengths)-np.pi*geometry.radius**2

    def field(self, xt):
        if not self.geometry.radius: return super().field(xt)
        enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            if not xt.requires_grad: xt = xt.detach().requires_grad_(True)
            xy = xt[:, :2]
            initial = self.geometry.initial_velocity(xy)
            r = self.fluid(self.features(xt))
            y, t = xt[:, 1:2], xt[:, -1:]
            b, u = self.geometry.factors(xy)
            velocity = initial+(t/self.cfg.final_time)*y*(1-y)*b*u*r[:, :2]
            # Same periodic pressure gauge, moved to a guaranteed-fluid point.
            anchor = torch.cat([torch.zeros_like(t), torch.full_like(t, .5), t], 1)
            pressure = r[:, 2:]-self.fluid(self.features(anchor))[:, 2:]
            field = torch.cat([velocity, pressure], 1)
        return field if enabled else field.detach()

    def residuals(self, t, xyz):
        if not self.geometry.fluid(xyz).all(): raise ValueError('PDE points must lie in fluid, not solid.')
        result = super().residuals(t, xyz)
        if self.geometry.radius:
            q, _ = self.particle_state(t)
            dx, y = self.geometry.dx(q[..., 0]), q[..., 1]
            lower = torch.relu(self.geometry.radius-torch.sqrt(dx*dx+y*y+1e-24))
            upper = torch.relu(self.geometry.radius-torch.sqrt(dx*dx+(1-y)**2+1e-24))
            result['confinement'] = result['confinement']+lower+upper
        return result


def train(model, folder, seconds):
    cfg = model.cfg; p = next(model.parameters())
    def batch(gen, audit=False):
        nt, ns = (max(7, cfg.time_batch), max(96, cfg.space_batch)) if audit else (cfg.time_batch, cfg.space_batch)
        t = torch.rand(nt, 1, generator=gen, dtype=p.dtype, device=p.device).square()*cfg.final_time
        if audit: t[0] = 0.; t[-1] = cfg.final_time
        xyz = model.geometry.sample(nt*ns, gen, dtype=p.dtype, device=p.device).reshape(nt, ns, 2)
        return t, xyz
    fixed = batch(torch.Generator(device=p.device).manual_seed(19481), audit=True)
    gen = torch.Generator(device=p.device).manual_seed(cfg.seed+101)
    def objective(audit=False): return model.loss(*(fixed if audit else batch(gen)))
    return fit(model, objective, cfg.epochs, cfg.lr, folder, max_seconds=seconds, audit_every=50,
               save_every_audit=True, decay=cfg.learning_rate_decay)
