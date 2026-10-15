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
import torch
from parcel_pinn.config import configuration
from parcel_pinn.model import CoupledPINN
from parcel_pinn.physics import derivative, interpolate, spread, weights
from parcel_pinn.training import seed_all
from gaussian_constriction.model import GeometrySettings, ConstrictedGaussianPINN, train

ROOT = Path(__file__).resolve().parents[1]


class GaussianConstrictionTests(unittest.TestCase):
    def setUp(self): seed_all(7)

    def model(self, radius=.25):
        c = configuration('smoke')
        return ConstrictedGaussianPINN(c, c.cases[0], GeometrySettings(radius=radius))

    def test_zero_radius_exactly_recovers_original_model_and_residuals(self):
        m = self.model(0); original = CoupledPINN(m.cfg, m.case); original.load_state_dict(m.state_dict())
        xt = torch.rand(57, 3, dtype=torch.float64); xt[:, 0] *= m.cfg.lengths[0]
        torch.testing.assert_close(m.field(xt), original.field(xt), rtol=0, atol=0)
        tt = torch.tensor([[.1], [.7]], dtype=torch.float64)
        xyz = torch.rand(2, 19, 2, dtype=torch.float64)*torch.tensor(m.cfg.lengths)
        for k, value in m.residuals(tt, xyz).items():
            torch.testing.assert_close(value, original.residuals(tt, xyz)[k], rtol=0, atol=0)
        self.assertEqual(m.physical, original.physical)
        torch.testing.assert_close(m.q0, original.q0, rtol=0, atol=0)

    def test_two_networks_and_original_physical_parameters_are_retained(self):
        m = self.model(); base = CoupledPINN(m.cfg, m.case)
        self.assertEqual(m.physical, base.physical)
        self.assertTrue(any(k.startswith('parcels.') for k in m.state_dict()))
        self.assertTrue(any(k.startswith('fluid.') for k in m.state_dict()))
        self.assertEqual(sum(p.numel() for p in m.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertTrue(m.geometry.fluid(m.quad).all())
        self.assertLess(len(m.quad), len(base.quad))
        self.assertEqual(m.initial_centers_resampled, 0)
        torch.testing.assert_close(m.q0, base.q0, rtol=0, atol=0)

    def test_curved_no_slip_periodicity_and_geometry_initial_continuity(self):
        m = self.model(); points, _, _ = m.geometry.boundary(65)
        for t in [0., .4, 1.]:
            xt = torch.tensor(np.column_stack([points, np.full(len(points), t)]), dtype=torch.float64)
            self.assertLess(float(m.field(xt)[:, :2].abs().max().detach()), 1e-11)
        left = torch.tensor([[0., .2, .4], [0., .5, .8], [0., .8, .3]], dtype=torch.float64)
        right = left.clone(); right[:, 0] = m.cfg.lengths[0]
        torch.testing.assert_close(m.field(left), m.field(right), atol=1e-12, rtol=1e-12)
        xy = m.geometry.sample(173, torch.Generator().manual_seed(377))
        xt = torch.cat([xy, torch.zeros(len(xy), 1, dtype=torch.float64)], 1).requires_grad_(True)
        field = m.field(xt)
        div = derivative(field[:, 0], xt)[:, 0]+derivative(field[:, 1], xt)[:, 1]
        self.assertLess(float(div.abs().max().detach()), 1e-10)

    def test_gaussian_normalization_force_and_work_on_fluid_quadrature(self):
        m = self.model()
        q = torch.tensor([[[m.geometry.xc, .5], [m.geometry.xc, .27], [.01, .6]]], dtype=torch.float64)
        W, _ = weights(m.quad[None], q, m.cfg.width, m.cfg.lengths, m.dv)
        u = torch.randn(1, len(m.quad), 2, dtype=torch.float64)
        force = torch.randn(1, 3, 2, dtype=torch.float64)
        sampled = interpolate(W, u, m.dv); reaction = spread(W, force)
        torch.testing.assert_close(W.sum(1)*m.dv, torch.ones(1, 3, dtype=torch.float64), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(reaction.sum(1)*m.dv, -force.sum(1), rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(float((reaction*u).sum()*m.dv), float(-(sampled*force).sum()), places=12)

    def test_solid_residual_points_rejected_and_curved_boundary_closes(self):
        m = self.model()
        with self.assertRaises(ValueError):
            m.residuals(torch.tensor([[.3]], dtype=torch.float64), torch.tensor([[[m.geometry.xc, .01]]], dtype=torch.float64))
        _, normal, ds = m.geometry.boundary(101)
        np.testing.assert_allclose((normal*ds[:, None]).sum(0), [0, 0], atol=1e-13)

    def test_float32_has_finite_gradients_through_both_networks(self):
        m = self.model().float(); gen = torch.Generator().manual_seed(42)
        xy = m.geometry.sample(38, gen, dtype=torch.float32).reshape(2, 19, 2)
        loss, _ = m.loss(torch.tensor([[.2], [.6]], dtype=torch.float32), xy); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for net in [m.fluid, m.parcels]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().max() > 0 for p in net.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_notebook_payload_is_current_and_calls_transient_gaussian_entry(self):
        nb = json.loads((ROOT/'BilateralConstriction_GPU.ipynb').read_text()); constants = {}; code = []
        for cell in nb['cells']:
            if cell['cell_type'] != 'code': continue
            source = ''.join(cell['source']); compile(source, '<notebook>', 'exec'); code.append(source)
            if 'SOURCE_B64 =' in source:
                for node in ast.parse(source).body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['SOURCE_B64', 'SOURCE_SHA256']:
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
        data = base64.b64decode(constants['SOURCE_B64'])
        self.assertEqual(hashlib.sha256(data).hexdigest(), constants['SOURCE_SHA256'])
        self.assertIn('--validation-pack', '\n'.join(code))
        self.assertIn('gaussian_constriction.model', '\n'.join(code))
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertIsNone(z.testzip()); self.assertIn('gaussian_constriction/model.py', z.namelist())
            self.assertNotIn('constricted_flow/model.py', z.namelist())
            for name in z.namelist(): self.assertEqual(z.read(name), (ROOT/name).read_bytes())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_real_cuda_training(self):
        m = self.model().float().cuda(); m.cfg.epochs = 2
        with tempfile.TemporaryDirectory() as td:
            status, _ = train(m, td, 30)
            self.assertFalse(status['termination'].startswith('training_error'))
            saved = torch.load(Path(td)/'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['state_dict'].values()))


if __name__ == '__main__': unittest.main()
