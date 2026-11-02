"""CPU admission, warm-start, feature-location and queue regression contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import numpy as np
from PIL import Image

from audit_full_data_protocol import experiment_tasks
from fullsize_training import extract_teacher_features_batched
from multisweep_teacher import MultiSweepTeacher, Downstream, load_official_camera_encoder
from run_official_full_data import ROOT, gpu_available, plan, validate_partial
from train_multisweep_teacher import model_inputs
from nuscenesdataset import img_transform


class OfficialFullDataTests(unittest.TestCase):
    def test_checkpoint_resize_is_opt_in_and_matches_pillow_default(self):
        image = Image.fromarray(np.random.default_rng(125).integers(0,256,(10,12,3),dtype=np.uint8))
        args = ((5,4), (0,0,5,4))
        self.assertTrue(np.array_equal(np.asarray(img_transform(image,*args)),
            np.asarray(image.resize((5,4),Image.Resampling.NEAREST))))
        self.assertTrue(np.array_equal(np.asarray(img_transform(image,*args,checkpoint_era_resize=True)),
            np.asarray(image.resize((5,4)))))

    def test_strict_encoder_only_initialization(self):
        encoder = nn.Linear(2, 3)
        source = nn.Linear(2, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'official.pth'
            state = {'encoder.'+k: v for k,v in source.state_dict().items()}
            state['decoder.not_loaded'] = torch.ones(1)
            torch.save(dict(model_state_dict=state), path)
            info = load_official_camera_encoder(encoder, path)
            self.assertEqual(info['tensors'], 2)
            self.assertTrue(torch.equal(encoder.weight, source.weight))
            del state['encoder.bias']
            torch.save(dict(model_state_dict=state), path)
            with self.assertRaises(RuntimeError):
                load_official_camera_encoder(encoder, path)

    def test_prefusion_pool_is_fixed_and_independent_of_radar(self):
        class Student(nn.Module):
            def forward(self, rgb_camXs, bevcar_voxels, **kwargs):
                camera = rgb_camXs.repeat_interleave(8, dim=1)
                return rgb_camXs + bevcar_voxels, camera, rgb_camXs
        model = MultiSweepTeacher(Student())
        camera = torch.randn(1, 128, 2, 3)
        inputs = (camera, None, None, None, torch.ones_like(camera))
        before, _, _ = model.features(inputs, 'camera_bev')
        after, _, _ = model.features(inputs, 'fused_bev')
        self.assertTrue(torch.allclose(before, camera))
        self.assertTrue(torch.allclose(after, camera+1))
        self.assertEqual(before.shape, after.shape)

    def test_zero_velocity_is_applied_at_shared_input_boundary(self):
        features = torch.ones(1, 2, 3, 7)
        prepared = (None, None, None, None, (features, torch.zeros(1,2,3), torch.tensor([2])))
        with patch('train_multisweep_teacher.prepare_bevcar_inputs', return_value=prepared):
            hidden = model_inputs(None, 'cpu', 'zero')[-1][0]
            self.assertEqual(float(hidden[..., 4:6].sum()), 0.)
            self.assertTrue(torch.equal(hidden[..., :4], features[..., :4]))
            self.assertTrue(torch.equal(hidden[..., 6], features[..., 6]))
            self.assertTrue(torch.equal(features, torch.ones_like(features)))

    def test_teacher_uses_requested_patch_grid(self):
        class Teacher:
            def forward_features(self, images):
                self.size = images.shape[-2:]
                n = (self.size[0]//14)*(self.size[1]//14)
                return dict(x_norm_patchtokens=torch.ones(len(images), n, 384))
        teacher = Teacher()
        result = extract_teacher_features_batched(teacher, torch.zeros(1,1,3,28,56),
                                                  'cpu', resolution=(28,42))
        self.assertEqual(teacher.size, (28,42))
        self.assertEqual(result.shape, (1,1,384,2,3))
        with self.assertRaises(ValueError):
            extract_teacher_features_batched(teacher, torch.zeros(1,1,3,28,56),
                                              'cpu', resolution=(28,43))

    def test_eleven_tasks_use_official_full_split(self):
        protocol = json.loads((ROOT/'configs/full_data_official_protocol.json').read_text())
        tasks = plan(Path('/tmp/study'), Path('/tmp/data'), Path('/tmp/bevcar'),
                     Path('/tmp/official.pth'), protocol)
        self.assertEqual(len(tasks), 11)
        for task in tasks:
            command = task['command']
            self.assertIn('--official-camera-checkpoint', command)
            self.assertNotIn('--train-samples', command)
            self.assertNotIn('--val-samples', command)
            start = command.index('--image-resolution')
            self.assertEqual(command[start+1:start+3], ['448','800'])
        for task in tasks[1:]:
            self.assertEqual(task['command'][task['command'].index('--velocity-mode')+1],
                             'zero' if 'velocity_zero' in task['name'] else 'full')
        self.assertTrue(all(t['status']=='queued' for t in experiment_tasks(True)))

    def test_busy_or_unqueryable_gpu_never_admits_training(self):
        with patch('run_official_full_data.subprocess.check_output', side_effect=['1234\n','23000, 0\n']):
            self.assertFalse(gpu_available()[0])
        with patch('run_official_full_data.subprocess.check_output', side_effect=['','23000, 0\n']):
            self.assertTrue(gpu_available()[0])
        with patch('run_official_full_data.subprocess.check_output', side_effect=OSError('no device')):
            self.assertFalse(gpu_available()[0])

    def test_partial_smoke_requires_real_finite_update(self):
        summary = dict(status='partial', progress=dict(update=3, latest=dict(loss=.3, gradient_norm=1.)))
        self.assertTrue(validate_partial(summary,3))
        self.assertFalse(validate_partial(summary,2))
        summary['progress']['latest']['loss'] = float('nan')
        self.assertFalse(validate_partial(summary,3))


if __name__ == '__main__':
    unittest.main()
