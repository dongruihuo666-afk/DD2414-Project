import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from dd2414.data.batch import NuScenesBatch, as_batch
from dd2414.interfaces import extract_features
from dd2414.cli.main import main
from dd2414.evaluation.segmentation import official_counts


class InterfaceTests(unittest.TestCase):
    def test_view_is_zero_copy_and_has_no_implicit_validation(self):
        values = [torch.randn(1) for _ in range(17)]
        view = as_batch(values)
        self.assertIs(view.images, values[0])
        self.assertIs(view.radar, values[16])
        self.assertIs(as_batch(view), view)
        self.assertEqual(values[16].data_ptr(), view.radar.data_ptr())
        with self.assertRaises(ValueError):
            view.validate()

    def test_full_shape_contract_and_rejected_camera_mismatch(self):
        values = [None] * 18
        values[:4] = [torch.zeros(2, 1, 6, 3, 4, 8), torch.zeros(2, 1, 6, 3, 3),
                      torch.zeros(2, 1, 6, 3), torch.zeros(2, 1, 6, 4, 4)]
        values[12:17] = [torch.zeros(2, 1, c, 5, 5) for c in (1, 1, 1, 2)] + [torch.zeros(2, 1, 19, 9)]
        self.assertIs(NuScenesBatch(values).validate().radar, values[16])
        values[3] = torch.zeros(2, 1, 5, 4, 4)
        with self.assertRaisesRegex(ValueError, 'intrinsics'):
            as_batch(values).validate()

    def test_feature_api_preserves_gradient_and_caller_policy(self):
        class Model(torch.nn.Module):
            def features(self, inputs, location):
                return inputs[0] * 2, inputs[0], None
        model = Model().train()
        value = torch.randn(3, requires_grad=True)
        output = extract_features(model, (value,), location='camera_bev')
        output.bev_features.sum().backward()
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(value.grad, torch.full_like(value, 2)))
        self.assertIs(output.image_features, value)

    def test_cli_help_is_stdlib_only(self):
        code = "from dd2414.cli.main import main; import sys; main(['train','--help'])"
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        # A fresh import guard makes accidental heavyweight help imports fail.
        code = "import sys; sys.modules['torch']=None; sys.modules['nuscenes']=None; " + code
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--nsweeps', result.stdout)

    def test_official_rounding_and_saved_array_evaluation(self):
        # Zero logit rounds sigmoid(.0)==.5 to background, matching Simple-BEV.
        logits = torch.tensor([[[[0., 4., -4., 4.]]]])
        target = torch.tensor([[[[1., 1., 0., 0.]]]])
        valid = torch.ones_like(target)
        self.assertEqual(official_counts(logits, target, valid), dict(tp=1, fp=1, fn=1, tn=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'values.npz'
            np.savez(path, logits=logits.numpy(), target=target.numpy(), valid=valid.numpy())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(['evaluate', '--arrays', str(path)])
            self.assertAlmostEqual(json.loads(output.getvalue())['vehicle_iou'], 1 / 3)
