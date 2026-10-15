import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from scipy.integrate import quad
import torch
from constricted_flow.model import Settings, Geometry, FlowPINN, train
from parcel_pinn.training import seed_all

ROOT = Path(__file__).resolve().parents[1]


class ConstrictedFlowTests(unittest.TestCase):
    def setUp(self): seed_all(7)

    def test_domain_area_sampling_and_throat(self):
        geo = Geometry()
        lo, hi = geo.bounds(np.array([0., geo.xc, geo.length]))
        np.testing.assert_allclose(lo, [0., .25, 0.]); np.testing.assert_allclose(hi, [1., .75, 1.])
        area = quad(lambda x: float(geo.bounds(np.array(x))[1]-geo.bounds(np.array(x))[0]),
                    0, geo.length, points=[geo.xc-geo.radius, geo.xc+geo.radius], epsabs=1e-10)[0]
        self.assertAlmostEqual(area, geo.length-np.pi*geo.radius**2, places=9)
        self.assertFalse(geo.fluid(np.array([[geo.xc, 0.], [geo.xc, 1.]])).any())
        for focused in [True, False]:
            points = geo.sample(4000, torch.Generator().manual_seed(71), focused=focused)
            self.assertTrue(geo.fluid(points).all())
            self.assertTrue(torch.isfinite(points).all())
            self.assertEqual(points.shape, (4000, 2))

    def test_closed_or_misplaced_geometry_is_rejected(self):
        for params in [dict(radius=.5), dict(radius=-.1), dict(center_fraction=.01),
                       dict(radius=float('nan')), dict(reynolds=[0])]:
            with self.assertRaises(ValueError): Settings(**params).validate()

    def test_hard_inlet_curved_walls_and_symmetry(self):
        cfg = Settings(hidden=16, depth=2)
        m = FlowPINN(cfg, cfg.cases()[1]); geo = m.geometry
        walls = torch.tensor(geo.walls(251), dtype=torch.float64)
        self.assertLess(float(m(walls)[:, :2].abs().max().detach()), 1e-13)
        y = torch.linspace(0, 1, 101, dtype=torch.float64)
        inlet = m(torch.stack([torch.zeros_like(y), y], 1))
        torch.testing.assert_close(inlet[:, 0], 6*y*(1-y), atol=1e-14, rtol=1e-14)
        self.assertLess(float(inlet[:, 1].abs().max().detach()), 1e-14)
        xy = geo.sample(79, torch.Generator().manual_seed(42)); reflected = xy.clone(); reflected[:, 1] = 1-xy[:, 1]
        a, b = m(xy), m(reflected)
        torch.testing.assert_close(a[:, [0, 2]], b[:, [0, 2]], atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(a[:, 1], -b[:, 1], atol=1e-13, rtol=1e-13)

    def test_straight_poiseuille_satisfies_shared_equations_outlet_and_flux(self):
        cfg = Settings(hidden=16, depth=2); m = FlowPINN(cfg, cfg.cases()[0])
        with torch.no_grad():
            for p in m.parameters(): p.zero_()
        xy = m.geometry.sample(131, torch.Generator().manual_seed(67))
        momentum, div = m.residuals(xy)
        self.assertLess(float(momentum.abs().max().detach()), 1e-13)
        self.assertLess(float(div.abs().max().detach()), 1e-13)
        xy[:, 0] = cfg.length
        self.assertLess(float(m.outlet(xy).abs().max().detach()), 1e-13)
        flux = m.flux(torch.linspace(0, cfg.length, 17, dtype=torch.float64))
        torch.testing.assert_close(flux, torch.ones_like(flux), atol=1e-13, rtol=1e-13)

    def test_float32_physics_gradients_and_fluid_only_parameter_tree(self):
        cfg = Settings(hidden=16, depth=2, flux_points=16)
        m = FlowPINN(cfg, cfg.cases()[1]).float(); geo = m.geometry
        xy = geo.sample(63, torch.Generator().manual_seed(55), dtype=torch.float32)
        outlet = xy[:13].clone(); outlet[:, 0] = cfg.length
        loss, parts = m.objective(xy, outlet, torch.linspace(0, cfg.length, 5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {'momentum', 'continuity', 'outlet', 'flux'})
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertFalse(any('parcel' in k for k in m.state_dict()))

    def test_curved_geometry_enforces_continuity_and_unit_flux_before_training(self):
        cfg = Settings(hidden=16, depth=2)
        m = FlowPINN(cfg, cfg.cases()[1])
        xy = m.geometry.sample(131, torch.Generator().manual_seed(720))
        _, div = m.residuals(xy)
        self.assertLess(float(div.abs().max().detach()), 1e-11)
        sections = torch.linspace(0, cfg.length, 61, dtype=torch.float64)
        with torch.no_grad(): flux = m.flux(sections, order=160)
        torch.testing.assert_close(flux, torch.ones_like(flux), atol=1e-8, rtol=1e-8)

    def test_standalone_notebook_payload_matches_source(self):
        nb = json.loads((ROOT/'BilateralConstriction_GPU.ipynb').read_text())
        constants = {}; code = []
        for c in nb['cells']:
            if c['cell_type'] != 'code': continue
            source = ''.join(c['source']); compile(source, '<notebook>', 'exec'); code.append(source)
            if 'SOURCE_B64 =' in source:
                for node in ast.parse(source).body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['SOURCE_B64', 'SOURCE_SHA256']:
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
        payload = base64.b64decode(constants['SOURCE_B64'])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), constants['SOURCE_SHA256'])
        self.assertNotIn('--validation-pack', '\n'.join(code))
        self.assertIn('run_constriction.py', '\n'.join(code))
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            self.assertIsNone(z.testzip())
            for name in z.namelist(): self.assertEqual(z.read(name), (ROOT/name).read_bytes())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable in this environment')
    def test_real_cuda_training(self):
        cfg = Settings(hidden=16, depth=2, interior_points=64, boundary_points=12, flux_points=16, epochs=2)
        m = FlowPINN(cfg, cfg.cases()[1]).float().cuda()
        with tempfile.TemporaryDirectory() as td:
            state, _ = train(m, td, 30)
            self.assertFalse(state['termination'].startswith('training_error'))
            saved = torch.load(Path(td)/'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.isfinite(t).all() for t in saved['state_dict'].values()))


if __name__ == '__main__': unittest.main()
