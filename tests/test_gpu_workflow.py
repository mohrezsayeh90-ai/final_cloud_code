import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from parcel_pinn.config import configuration
from parcel_pinn.gpu_workflow import supervised
from parcel_pinn.model import CoupledPINN
from parcel_pinn.training import fit, seed_all, train_channel


ROOT = Path(__file__).resolve().parents[1]


class GPUWorkflowTests(unittest.TestCase):
    def test_float32_and_float64_residuals_and_gradients_agree(self):
        seed_all(7)
        cfg = configuration('smoke')
        double = CoupledPINN(cfg, cfg.cases[0])
        single = CoupledPINN(cfg, cfg.cases[0]).float()
        single.load_state_dict(double.state_dict())
        t = torch.tensor([[.1], [.4]], dtype=torch.float64)
        xyz = torch.rand(2, 13, 2, dtype=torch.float64)*torch.tensor(cfg.lengths)
        a, _ = double.loss(t, xyz)
        b, _ = single.loss(t.float(), xyz.float())
        torch.testing.assert_close(a.detach().float(), b.detach(), rtol=2e-5, atol=2e-5)
        a.backward(); b.backward()
        for p, q in zip(double.parameters(), single.parameters()):
            torch.testing.assert_close(p.grad.float(), q.grad, rtol=2e-3, atol=2e-4)

    def test_deadline_preserves_checked_checkpoint(self):
        model = torch.nn.Linear(1, 1).double()
        x = torch.ones(3, 1, dtype=torch.float64)
        def objective(audit=False):
            loss = (model(x)-2).square().mean()
            return loss, {'residual': loss}
        with tempfile.TemporaryDirectory() as td:
            # Simulated clock expires after the initial audit, before an optimizer step.
            with patch('parcel_pinn.training.time.monotonic', side_effect=range(100)):
                state, _ = fit(model, objective, 10000, .01, td, max_seconds=.5,
                               audit_every=1, save_every_audit=True)
            self.assertEqual(state['termination'], 'time_budget_exhausted')
            self.assertEqual(state['best_step'], 0)
            checkpoint = torch.load(Path(td)/'checkpoint.pt', weights_only=True)
            self.assertEqual(checkpoint['training']['checkpoint_kind'], 'initialized')
            self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values()))

    def test_supervisor_stops_worker_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td)/'worker.log'
            result = supervised([sys.executable, '-c',
                                 'import time; print("started", flush=True); time.sleep(30)'], .7, log)
            self.assertEqual(result['outcome'], 'timeout')
            self.assertIn('started', log.read_text())
            self.assertLess(result['seconds'], 5)

    def test_notebook_embeds_current_source_and_compiles(self):
        notebook = json.loads((ROOT/'GaussianParcelPINN_GPU.ipynb').read_text())
        scope = {}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, '<notebook>', 'exec')
                if 'SOURCE_B64 =' in source:
                    # Read the two literal assignments without extracting into the repo.
                    import ast
                    for node in ast.parse(source).body:
                        if isinstance(node, ast.Assign) and node.targets[0].id in ('SOURCE_B64', 'SOURCE_SHA256'):
                            scope[node.targets[0].id] = ast.literal_eval(node.value)
        payload = base64.b64decode(scope['SOURCE_B64'])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), scope['SOURCE_SHA256'])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT/name).read_bytes(), name)
            self.assertIn('parcel_pinn/gpu_workflow.py', archive.namelist())
            self.assertIn('reference_inputs/Validation_Pack/data/analytic_stokes.csv', archive.namelist())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable in this environment')
    def test_real_cuda_training_and_cpu_export(self):
        seed_all(7)
        cfg = configuration('smoke'); cfg.epochs = 2
        model = CoupledPINN(cfg, cfg.cases[0]).float().cuda()
        with tempfile.TemporaryDirectory() as td:
            state, _ = train_channel(model, td, max_seconds=30, audit_every=1, save_every_audit=True)
            self.assertFalse(state['termination'].startswith('training_error'))
            self.assertGreater(state['best_step'], 0)
            target = CoupledPINN(cfg, cfg.cases[0])
            target.load_state_dict(torch.load(Path(td)/'checkpoint.pt', map_location='cpu', weights_only=True)['state_dict'])
            self.assertEqual(next(target.parameters()).device.type, 'cpu')
            self.assertTrue(torch.isfinite(target.field(torch.rand(8, 3, dtype=torch.float64))).all())


if __name__ == '__main__':
    unittest.main()
