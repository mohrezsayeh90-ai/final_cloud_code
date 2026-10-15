from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from parcel_pinn.model import MLP
from parcel_pinn.physics import derivative


@dataclass
class Settings:
    length: float = 2.5  # H=1, corresponding to L=1 and H=.4 in the sketch
    center_fraction: float = .4
    radius: float = .25
    reynolds: tuple = (10., 20., 50.)
    hidden: int = 48
    depth: int = 4
    interior_points: int = 512
    boundary_points: int = 48
    flux_points: int = 48
    epochs: int = 1000000
    lr: float = .001
    seed: int = 7
    threads: int = 2
    dpi: int = 160
    frames: int = 31
    residual_tolerance: float = .05
    flux_tolerance: float = .05
    outlet_tolerance: float = .05

    def validate(self):
        scalars = [self.length, self.center_fraction, self.radius, self.lr,
                   self.residual_tolerance, self.flux_tolerance, self.outlet_tolerance]
        if not np.isfinite(scalars).all() or self.length <= 0 or not 0 < self.center_fraction < 1:
            raise ValueError('Use finite scales, positive L/H and a center inside the channel.')
        xc = self.center_fraction*self.length
        if not 0 <= self.radius < .5 or self.radius >= min(xc, self.length-xc):
            raise ValueError('Need 0 <= r/H < .5 and obstacles clear of inlet and outlet.')
        if not self.reynolds or any(not np.isfinite(r) or r <= 0 for r in self.reynolds):
            raise ValueError('Reynolds numbers must be positive and finite.')
        if len(set(self.reynolds)) != len(self.reynolds):
            raise ValueError('Reynolds numbers must be unique.')
        if min(self.hidden, self.depth, self.interior_points, self.boundary_points,
               self.flux_points, self.threads, self.frames, self.dpi) < 2 or self.epochs < 0 or self.lr <= 0:
            raise ValueError('Invalid training/sampling settings.')
        if min(self.residual_tolerance, self.flux_tolerance, self.outlet_tolerance) <= 0:
            raise ValueError('Screening tolerances must be positive.')
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def cases(self):
        return [dict(name='straight_control', re=20., radius=0.)]+[
            dict(name=f'constricted_Re{r:g}'.replace('.', 'p'), re=r, radius=self.radius)
            for r in self.reynolds]


class Geometry:
    def __init__(self, length=2.5, center_fraction=.4, radius=.25):
        Settings(length=length, center_fraction=center_fraction, radius=radius).validate()
        self.length, self.xc, self.radius = length, center_fraction*length, radius

    def fluid(self, xy, margin=0.):
        x, y = xy[..., 0], xy[..., 1]
        valid = (x >= 0) & (x <= self.length) & (y >= 0) & (y <= 1)
        if self.radius:
            r = self.radius+margin
            valid = valid & ((x-self.xc)**2+y**2 >= r*r)
            valid = valid & ((x-self.xc)**2+(y-1)**2 >= r*r)
        return valid

    def bounds(self, x):
        """Cross-section limits for plotting/flux integration, NOT differentiated."""
        x = np.asarray(x)
        lower = np.sqrt(np.maximum(self.radius**2-(x-self.xc)**2, 0))
        return lower, 1-lower

    def sample(self, n, generator, *, dtype=torch.float64, device='cpu', focused=False):
        parts, count = [], 0
        while count < n:
            points = torch.rand(max(64, 2*(n-count)), 2, generator=generator, dtype=dtype, device=device)
            points[:, 0] *= self.length
            if focused and self.radius:
                near = torch.arange(len(points), device=device) % 2 == 0
                points[near, 0] = self.xc+(points[near, 0]/self.length-.5)*4*self.radius
            accepted = points[self.fluid(points, margin=1e-6)]
            parts.append(accepted); count += len(accepted)
        return torch.cat(parts)[:n]

    def walls(self, n=101):
        x = np.linspace(0, self.length, n)
        outside = np.abs(x-self.xc) >= self.radius
        pieces = [np.column_stack([x[outside], np.zeros(sum(outside))]),
                  np.column_stack([x[outside], np.ones(sum(outside))])]
        if self.radius:
            angle = np.linspace(0, np.pi, n)
            pieces.extend([np.column_stack([self.xc+self.radius*np.cos(angle), self.radius*np.sin(angle)]),
                           np.column_stack([self.xc+self.radius*np.cos(angle), 1-self.radius*np.sin(angle)])])
        return np.concatenate(pieces)


def steady_terms(field, xy, nu):
    gradients = [derivative(field[:, k], xy) for k in range(3)]
    lap = torch.stack([sum(derivative(gradients[k][:, j], xy)[:, j] for j in range(2))
                       for k in range(2)], 1)
    adv = torch.stack([(field[:, :2]*gradients[k]).sum(1) for k in range(2)], 1)
    momentum = adv+gradients[2]-nu*lap
    div = gradients[0][:, 0]+gradients[1][:, 1]
    return momentum, div, gradients


class FlowPINN(nn.Module):
    def __init__(self, cfg, case):
        super().__init__()
        self.cfg, self.case = cfg, case
        self.geometry = Geometry(cfg.length, cfg.center_fraction, case['radius'])
        self.net = MLP(4, 2, cfg.hidden, cfg.depth).double()

    def wall_coordinate(self, x, y):
        """Smooth fluid coordinate: s=0/1 on walls and s=y, s_x=0 at inlet.

        Circle level sets avoid differentiating the square-root wall graph.
        That graph has vertical tangents at the semicircle endpoints, including
        along vertical extensions into the interior when used as a map.
        """
        r = self.geometry.radius
        if not r:
            return y
        d1 = (x-self.geometry.xc)**2+y*y-r*r
        d2 = (x-self.geometry.xc)**2+(1-y)**2-r*r
        regularizer = r*r*(x/self.geometry.xc)**2
        bottom = y*d1/(d1+regularizer)
        top = (1-y)*d2/(d2+regularizer)
        return bottom/(bottom+top)

    def potentials(self, xy):
        x, y = xy[:, :1], xy[:, 1:2]
        L, re = self.cfg.length, self.case['re']
        symmetry = 2*y-1
        scale = max(self.geometry.radius, .15)
        local = torch.exp(-((x-self.geometry.xc)/scale)**2)
        features = torch.cat([2*x/L-1, symmetry.square(), local, local*symmetry.square()], 1)
        raw = self.net(features)
        s = self.wall_coordinate(x, y)
        # psi=0 at lower wall and psi=1 at upper wall. The value AND normal
        # derivative of the correction vanish on both boundaries and at inlet.
        psi = 3*s*s-2*s*s*s+8*(x/L)**2*(s*(1-s))**2*symmetry*raw[:, :1]
        p = 12/re*(L-x)+raw[:, 1:2]
        return psi, p

    def forward(self, xy):
        # A streamfunction needs one derivative even for inference. Honor an
        # outer no_grad context by detaching only AFTER forming the velocity.
        need_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if not xy.requires_grad: xy = xy.detach().requires_grad_(True)
            psi, pressure = self.potentials(xy)
            dpsi = derivative(psi[:, 0], xy)
            field = torch.cat([dpsi[:, 1:2], -dpsi[:, :1], pressure], 1)
        return field if need_graph else field.detach()

    def residuals(self, xy):
        xy = xy.detach().requires_grad_(True)
        return steady_terms(self(xy), xy, 1/self.case['re'])[:2]

    def outlet(self, xy):
        xy = xy.detach().requires_grad_(True)
        field = self(xy)
        du, dv = derivative(field[:, 0], xy), derivative(field[:, 1], xy)
        # Natural open outlet of the LAPLACIAN weak form: nu*d_n(u)-p*n=0.
        # This is the do-nothing condition, not zero symmetric Cauchy traction.
        # It admits the exact fully developed straight-channel Poiseuille field.
        return torch.stack([-field[:, 2]+du[:, 0]/self.case['re'],
                            dv[:, 0]/self.case['re']], 1)

    def flux(self, positions, order=None):
        # Flux quadrature follows the exact semicircle sections; nodes are held fixed.
        nodes, weights = np.polynomial.legendre.leggauss(order or self.cfg.flux_points)
        lo, hi = self.geometry.bounds(positions.detach().cpu().numpy())
        a, b = positions.new_tensor(lo), positions.new_tensor(hi)
        y = a[:, None]+(positions.new_tensor(nodes)[None]+1)*(b-a)[:, None]/2
        xy = torch.stack([positions[:, None].expand_as(y), y], -1)
        velocity = self(xy.reshape(-1, 2))[:, 0].reshape_as(y)
        return (velocity*positions.new_tensor(weights)[None]).sum(1)*(b-a)/2

    def objective(self, interior, outlet, sections):
        momentum, div = self.residuals(interior)
        traction = self.outlet(outlet)
        flux = self.flux(sections)-1
        terms = dict(momentum=momentum.square().mean(), continuity=div.square().mean(),
                     outlet=traction.square().mean(), flux=flux.square().mean())
        return terms['momentum']+5*terms['continuity']+5*terms['outlet']+10*terms['flux'], terms


def train(model, folder, seconds):
    from parcel_pinn.training import fit
    p = next(model.parameters()); cfg, geo = model.cfg, model.geometry
    gen = torch.Generator(device=p.device).manual_seed(cfg.seed+104)
    def batch(generator, focused=True):
        interior = geo.sample(cfg.interior_points, generator, dtype=p.dtype, device=p.device, focused=focused)
        yy = torch.rand(cfg.boundary_points, generator=generator, dtype=p.dtype, device=p.device)
        outlet = torch.stack([torch.full_like(yy, cfg.length), yy], 1)
        sections = torch.linspace(0, cfg.length, 11, device=p.device, dtype=p.dtype)
        sections = torch.cat([sections, sections.new_tensor([geo.xc])])
        return interior, outlet, sections
    audit = batch(torch.Generator(device=p.device).manual_seed(9401), focused=False)
    def objective(audit=False):
        return model.objective(*(fixed if audit else batch(gen)))
    fixed = audit
    return fit(model, objective, cfg.epochs, cfg.lr, folder, max_seconds=seconds,
               audit_every=50, save_every_audit=True)
