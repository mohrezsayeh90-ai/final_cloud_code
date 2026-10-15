"""V01--V10 in atlas order. Analytical arrays are held out of training."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import chi2
import torch
from torch import nn
from .model import MLP
from .physics import (binary_impact, derivative, flow_terms, grid, group_event_count,
                      interpolate, sample_impact_normal, spread, wall_reflection, weights)
from .training import fit, seed_all


def metrics(pred, ref):
    p, r = np.asarray(pred), np.asarray(ref)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('metrics require finite matching arrays')
    den = np.linalg.norm(r)
    return dict(rmse=float(np.sqrt(np.mean((p-r)**2))),
                max_error=float(np.max(np.abs(p-r))),
                relative_l2=float(np.linalg.norm(p-r)/den) if den > 1e-14 else None)


class ResponsePINN(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.end = 5. if kind == 'V01' else 2.
        self.net = MLP(1, 2, cfg.hidden, cfg.depth).double()

    def forward(self, t):
        a = self.net(2*t/self.end-1)
        if self.kind == 'V01':
            return torch.cat([t*a[:, :1], t*t*a[:, 1:2]], 1)
        # Hard IC and total momentum. The PDE still determines the decay.
        u = t*a[:, :1]
        return torch.cat([u, 1-4*u], 1)

    def residual(self, t):
        t = t.detach().requires_grad_(True)
        y = self(t)
        dy = torch.cat([derivative(y[:, k], t) for k in range(2)], 1)
        if self.kind == 'V01':
            return torch.cat([dy[:, :1]-(1-y[:, :1]), dy[:, 1:2]-y[:, :1]], 1)
        rate = (y[:, :1]-y[:, 1:2])/.2
        return torch.cat([dy[:, :1]+.25*rate, dy[:, 1:2]-rate], 1)


class TaylorGreenPINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.end = 5.
        self.net = MLP(5, 3, cfg.hidden, cfg.depth).double()

    def forward(self, xt):
        x, y, t = xt.unbind(1)
        features = torch.stack([x.sin(), x.cos(), y.sin(), y.cos(), 2*t/self.end-1], 1)
        r = self.net(features)
        # Only initial conditions are supplied, no analytic temporal decay.
        u = x.sin()*y.cos()+t/self.end*r[:, 0]
        v = -x.cos()*y.sin()+t/self.end*r[:, 1]
        anchor = torch.stack([torch.zeros_like(t), torch.ones_like(t),
                              torch.zeros_like(t), torch.ones_like(t), 2*t/self.end-1], 1)
        gauge = self.net(anchor)[:, 2]
        p = .25*((2*x).cos()+(2*y).cos()-2)+t/self.end*(r[:, 2]-gauge)
        return torch.stack([u, v, p], 1)

    def residual(self, xt):
        xt = xt.detach().requires_grad_(True)
        r, div, _ = flow_terms(self(xt), xt, .01)
        return torch.cat([r, div[:, None]], 1)


def audit_catalog(pack, out):
    """Visit every literature case, preserving catalogue order and source metadata."""
    cases = json.loads((pack/'case_catalog.json').read_text())
    records = []
    for order, case in enumerate(cases, 1):
        family = case['case_id'][:2]
        if family == 'ER':
            reason = 'ERPP diffusion regularization, self-disturbance correction and exact source setup are not implemented; V01/V02/V03 are reduced verification only.'
        elif family == 'AP':
            reason = 'Finite-volume displacement and dense contact closure required; dilute NS is incompatible.'
        elif family in ('CO', 'DK', 'WK', 'GS'):
            reason = 'Resolved 2D cylinder, finite-size hydrodynamic forces/torques and matching boundaries required.'
        elif family == 'OB':
            reason = 'Spheroid orientation, anisotropic hydrodynamics and torque are absent.'
        elif family == 'HG':
            reason = 'Raw experimental arrays not bundled; wall lubrication, finite-size and wet impact physics absent.'
        elif family == 'TC':
            reason = 'Confined finite sphere, wall/history effects and digitized measured curves required.'
        else:
            reason = 'Resolved finite-size sphere and matching force/torque, boundary and wake physics required.'
        records.append(dict(order=order, case_id=case['case_id'], family=case['family'],
                            readiness=case['readiness'], outcome='not_run',
                            reason=reason, missing=case.get('missing', ''),
                            reference=case.get('data', ''), source=case.get('article_url', ''),
                            source_locator=case.get('source_locator', '')))
    pd.DataFrame(records).to_csv(out/'literature_audit.csv', index=False)
    return records


def verify(cfg, pack, out, plot):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    records = []
    def add(case_id, ok, details, **extras):
        records.append(dict(case_id=case_id, outcome='passed' if ok else 'failed',
                            evidence='numerical_verification', details=details, **extras))
    # Continue remaining checks after an individual failure, preserving diagnostics.
    for kind in ['V01', 'V02']:
        try:
            seed_all(cfg.seed, cfg.threads)
            model = ResponsePINN(cfg, kind)
            gen = torch.Generator().manual_seed(330)
            fixed = torch.linspace(0, model.end, 101, dtype=torch.float64)[:, None]
            def objective(audit=False):
                t = fixed if audit else torch.rand(96, 1, generator=gen, dtype=torch.float64)*model.end
                r = model.residual(t)
                loss = (r*r).mean()
                return loss, {'ode': loss}
            polish_t=torch.linspace(0,model.end,257,dtype=torch.float64)[:,None]
            def polish():
                r=model.residual(polish_t)
                return r.square().mean(),{}
            status, history = fit(model, objective, cfg.ode_epochs, cfg.lr, out/kind,
                                  polish_steps=cfg.ode_lbfgs_steps,polish_function=polish)
            filename = 'analytic_stokes.csv' if kind == 'V01' else 'analytic_two_way_relaxation.csv'
            reference = pd.read_csv(pack/'data'/filename)
            t = reference.iloc[:, 0].to_numpy()
            pred = model(torch.tensor(t[:, None], dtype=torch.float64)).detach().numpy()
            cols = list(reference.columns[1:3])
            table = pd.DataFrame({'time': t})
            ms = {}
            for k, name in enumerate(cols):
                table[name+'_reference'] = reference[name]
                table[name+'_pinn'] = pred[:, k]
                ms[name] = metrics(pred[:, k], reference[name])
            if kind == 'V02':
                table['total_momentum_pinn']=pred[:,0]+.25*pred[:,1]
                table['total_momentum_reference']=reference.total_momentum
                table['kinetic_energy_pinn']=.5*pred[:,0]**2+.125*pred[:,1]**2
                ms['total_momentum']=metrics(table.total_momentum_pinn,reference.total_momentum)
            table.to_csv(out/kind/'comparison.csv', index=False)
            independent = torch.tensor(np.random.default_rng(98).uniform(0, model.end, (211, 1)), dtype=torch.float64)
            residual = float(model.residual(independent).square().mean().sqrt().detach())
            good = residual < cfg.residual_tolerance and all(m['relative_l2'] < cfg.relative_l2_tolerance for m in ms.values())
            add(kind, good, 'Physics-only ODE PINN; supplied analytical CSV held out.',
                metrics=ms, independent_residual_rms=residual, training=status,
                tolerances=dict(relative_l2=cfg.relative_l2_tolerance, residual_rms=cfg.residual_tolerance))
            plot('response', table, cols, kind, out)
        except Exception as exc:
            records.append(dict(case_id=kind, outcome='error', details=f'{type(exc).__name__}: {exc}'))

    try:
        # V03 uses the requested 2D unit square, and the same transfer implementation.
        xyz, dv = grid((1., 1.), (48, 48))
        centers = torch.tensor([[[.01, .5], [.99, .5], [1.01, .5], [.5, .01], [.5, .99]]], dtype=torch.float64)
        W, _ = weights(xyz[None], centers, .1, (1., 1.), dv, periodic=(0,))
        norm = (W.sum(1)*dv).numpy().ravel()
        const = interpolate(W, torch.ones(1, len(xyz), 2, dtype=torch.float64)*2.7, dv).numpy()
        err = max(float(np.max(np.abs(norm-1))), float(np.max(np.abs(const-2.7))))
        add('V03', err < 1e-12, '2D wall and periodic-seam kernels; constant field sampling.', max_error=err)
        pd.DataFrame({'x': centers[0, :, 0], 'y': centers[0, :, 1], 'integral': norm}).to_csv(out/'V03.csv', index=False)
    except Exception as exc:
        records.append(dict(case_id='V03', outcome='error', details=str(exc)))
    try:
        ref = pd.read_csv(pack/'data'/'analytic_gaussian_overlap3d.csv')
        # Independent numerical integration in each Cartesian direction, not the overlap formula.
        def product_integral(separation):
            def fn(x):
                return math.exp(-x*x/(2*.5**2)-(x-separation)**2/(2*.75**2))/(2*math.pi*.5*.75)
            return quad(fn, -10, 10, epsabs=1e-13, epsrel=1e-11)[0]
        trans = product_integral(0)**2
        values = np.array([product_integral(r)*trans for r in ref.distance])
        ms = metrics(values, ref.overlap)
        add('V04', ms['relative_l2'] < 1e-8, 'Independent 3D product quadrature against supplied free-space array.', metrics=ms)
        frame = pd.DataFrame({'time': ref.distance, 'overlap_reference': ref.overlap, 'overlap_pinn': values})
        frame.to_csv(out/'V04.csv', index=False)
        plot('overlap', frame, ['overlap'], 'V04', out)
    except Exception as exc:
        records.append(dict(case_id='V04', outcome='error', details=str(exc)))
    for kind, a, b, e, ea, eb in [
        ('V05', [1., 0.], [-1., 0.], .8, [-.8, 0.], [.8, 0.]),
        ('V06', [1., .5], [-1., -.5], 1., [-1., .5], [1., -.5])]:
        va, vb = binary_impact(a, b, 1., 1., [-1., 0.], e)
        err = float(max(np.max(np.abs(va-ea)), np.max(np.abs(vb-eb))))
        add(kind, err < 1e-12, 'Frictionless physical binary impact algebra; not a channel collision validation.',
            max_error=err, velocity_1=va.tolist(), velocity_2=vb.tolist(),
            final_energy=float((va@va+vb@vb)/2))
    v = wall_reflection([2., -3.], [0., 1.], .6)
    add('V07', np.allclose(v, [2., 1.8]), 'Physical wall restitution algebra.', outgoing_velocity=v.tolist())
    try:
        rng = np.random.default_rng(cfg.seed)
        overlap = math.exp(-.005**2/(4*.005**2))/(4*math.pi*.005**2)**1.5
        K = math.pi*(2e-4)**2*.01
        N, dt = 1000, .001
        rate = N*K*overlap
        expected_group = rate*dt
        repetitions = 500000
        counts = group_event_count(rate, np.full(repetitions, dt), rng)
        total = counts.sum()
        lo = .5*chi2.ppf(.025, 2*total)/repetitions*N if total else 0.
        hi = .5*chi2.ppf(.975, 2*(total+1))/repetitions*N
        expected_physical = N*expected_group
        # Predetermined five-standard-error criterion; CI reported independently.
        stderr = N*math.sqrt(expected_group/repetitions)
        observed = float(counts.mean()*N)
        normals = np.array([sample_impact_normal([1, 0, 0], rng) for _ in range(10000)])
        angular_error = abs(float(np.mean(normals[:, 0]**2))-.5)
        add('V08', abs(observed-expected_physical) < 5*stderr and angular_error < .02,
            'Frozen-rate Poisson group counts and hard-sphere impact-normal sampler. Does not verify an evolving collision gas.',
            repetitions=repetitions, expected_physical_count=expected_physical, observed_physical_count=observed,
            mean_ci95=[lo, hi], five_standard_errors=5*stderr, expected_group_count=expected_group,
            impact_normal_second_moment_error=angular_error)
        pd.DataFrame({'quantity': ['expected', 'observed', 'ci_low', 'ci_high'],
                      'physical_count': [expected_physical, observed, lo, hi]}).to_csv(out/'V08.csv', index=False)
    except Exception as exc:
        records.append(dict(case_id='V08', outcome='error', details=str(exc)))
    try:
        xyz, dv = grid((1., 1., 1.), (16, 16, 16))
        rng = np.random.default_rng(42)
        parents = torch.tensor(rng.uniform(.15, .85, (1, 10, 3)), dtype=torch.float64)
        slip = torch.tensor(rng.normal(size=(1, 10, 3)), dtype=torch.float64)
        deposits, totals = [], []
        for count in [10, 50, 100]:
            factor = count//10
            q = parents.repeat_interleave(factor, dim=1)
            W, _ = weights(xyz[None], q, .1, (1., 1., 1.), dv)
            f = slip.repeat_interleave(factor, dim=1)*(10000/count)*1e-5
            deposits.append(spread(W, f).numpy())
            totals.append(count*(10000/count))
        err = max(float(np.max(np.abs(a-deposits[0]))) for a in deposits)
        add('V09', err < 1e-11 and totals == [10000]*3,
            'Coincident 10/50/100-parcel splitting at fixed population 10000. Spatial/statistical refinement remains separate.',
            max_force_density_difference=err, physical_counts=totals)
        pd.DataFrame({'parcels': [10, 50, 100], 'population': totals,
                      'max_difference': [float(np.max(np.abs(a-deposits[0]))) for a in deposits]}).to_csv(out/'V09.csv', index=False)
    except Exception as exc:
        records.append(dict(case_id='V09', outcome='error', details=str(exc)))
    try:
        seed_all(cfg.seed, cfg.threads)
        model = TaylorGreenPINN(cfg)
        gen = torch.Generator().manual_seed(142)
        def sample(gen, n):
            return torch.rand(n, 3, generator=gen, dtype=torch.float64)*torch.tensor([2*math.pi, 2*math.pi, model.end])
        audit = sample(torch.Generator().manual_seed(829), 128)
        def objective(audit=False):
            r = model.residual(fixed if audit else sample(gen, 128))
            return r.square().mean(), {'NS': r.square().mean()}
        fixed = audit
        polish_points=sample(torch.Generator().manual_seed(8193),384)
        def polish():
            r=model.residual(polish_points)
            return r.square().mean(),{}
        status, history = fit(model, objective, cfg.tg_epochs, cfg.lr, out/'V10',
                              polish_steps=cfg.tg_lbfgs_steps,polish_function=polish)
        points = sample(torch.Generator().manual_seed(391), 768)
        x, y, t = points.numpy().T
        reference = np.stack([np.sin(x)*np.cos(y)*np.exp(-.02*t),
                              -np.cos(x)*np.sin(y)*np.exp(-.02*t),
                              .25*(np.cos(2*x)+np.cos(2*y)-2)*np.exp(-.04*t)], 1)
        pred = model(points).detach().numpy()
        ms = {key: metrics(pred[:, k], reference[:, k]) for k, key in enumerate(['u', 'v', 'p'])}
        residual = float(model.residual(points[:192]).square().mean().sqrt().detach())
        good = residual < cfg.residual_tolerance and all(m['relative_l2'] < cfg.relative_l2_tolerance for m in ms.values())
        add('V10', good, 'Periodic 2D Taylor–Green PINN, nu=.01, t in [0,5], pressure gauge p(0,0,t)=0; analytic solution held out.',
            metrics=ms, independent_residual_rms=residual, training=status,
            tolerances=dict(relative_l2=cfg.relative_l2_tolerance, residual_rms=cfg.residual_tolerance))
        data = pd.DataFrame(dict(x=x, y=y, time=t, u_reference=reference[:, 0], u_pinn=pred[:, 0],
                                 v_reference=reference[:, 1], v_pinn=pred[:, 1], p_reference=reference[:, 2], p_pinn=pred[:, 2]))
        data.to_csv(out/'V10'/'comparison.csv', index=False)
        plot('taylor', data, ['u', 'v', 'p'], 'V10', out)
    except Exception as exc:
        records.append(dict(case_id='V10', outcome='error', details=f'{type(exc).__name__}: {exc}'))
    (out/'verification.json').write_text(json.dumps(records, indent=2))
    pd.DataFrame([{k: r.get(k) for k in ['case_id', 'outcome', 'details', 'independent_residual_rms']} for r in records]).to_csv(out/'verification.csv', index=False)
    return records
