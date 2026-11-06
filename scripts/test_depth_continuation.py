"""CPU contracts for an opt-in, fresh-optimizer depth continuation phase."""

import copy
from pathlib import Path
import unittest

import torch
from torch import nn

from multisweep_teacher import MultiSweepTeacher, relative_depth_target
from source_provenance import check_continuation_source


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.initialization = dict(path='/cluster/official_rgb.pth', sha256='rgb-hash',
            tensors=568, scope='encoder.* only; no decoder/fusion weights')
        self.config = dict(experiment='baseline_multisweep_double_distillation',
            mode='pretrain', depth_mode='none', epochs=5, nsweeps=10, seed=125,
            image_resolution=[448,800], teacher_resolution=[448,798], batch_size=1,
            train_samples=28130, val_samples=6019, image_weight=.2, bev_weight=.8,
            motion_weight=.5, precision='bf16', velocity_mode='full',
            manifest_sha256='manifest-hash', source_code_commit='source-commit',
            official_camera_checkpoint='/workstation/official_rgb.pth',
            official_camera_initialization=dict(self.initialization,
                path='/workstation/official_rgb.pth'))
        self.source = dict(run_config=self.config, sampler=dict(epoch=5),
            progress=dict(epoch=5, update=140650, validations={'5':dict(samples=6019)}))

    def check(self, source=None, expected=None, relocate=True):
        return check_continuation_source(source or self.source,
            expected or dict(velocity_mode='full'), 'manifest-hash',
            Path('/cluster/official_rgb.pth'), self.initialization, relocate)

    def test_completed_source_starts_new_phase_not_exact_resume(self):
        result = self.check()
        self.assertEqual(result['source_epochs'], 5)
        self.assertEqual(result['source_velocity_mode'], 'full')
        self.assertFalse(result['optimizer_restored'])
        self.assertTrue(result['sampler_restarted'])

    def test_partial_or_supervised_parent_is_rejected(self):
        for field, value in [('mode','finetune'),('depth_mode','radar'),('epochs',4)]:
            source = copy.deepcopy(self.source)
            source['run_config'][field] = value
            with self.assertRaises(ValueError):
                self.check(source)
        source = copy.deepcopy(self.source)
        source['progress']['validations'] = {}
        with self.assertRaises(ValueError):
            self.check(source)
        for section, field, value in [('sampler','epoch',4),('progress','epoch',4),
                                       ('progress','update',140649)]:
            source = copy.deepcopy(self.source)
            source[section][field] = value
            with self.assertRaises(ValueError):
                self.check(source)

    def test_velocity_treatment_and_scientific_configuration_stay_matched(self):
        for field,value in [('velocity_mode','zero'),('seed',126),('nsweeps',5),
                            ('image_resolution',[112,192]),('train_samples',4096),
                            ('batch_size',2),('manifest_sha256','other')]:
            source = copy.deepcopy(self.source)
            source['run_config'][field] = value
            with self.assertRaises(ValueError):
                self.check(source)

    def test_hidden_velocity_parent_requires_explicit_matching_treatment(self):
        source = copy.deepcopy(self.source)
        source['run_config']['velocity_mode'] = 'zero'
        result = self.check(source, dict(velocity_mode='zero'))
        self.assertEqual(result['source_velocity_mode'], 'zero')

    def test_rgb_relocation_does_not_relax_content_validation(self):
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):
            self.check(relocate=False)
        source = copy.deepcopy(self.source)
        source['run_config']['official_camera_initialization']['sha256'] = 'other'
        with self.assertRaisesRegex(ValueError,'content/provenance mismatch'):
            self.check(source)

    def test_all_model_tensors_load_strictly_into_each_depth_treatment(self):
        original = MultiSweepTeacher(nn.Linear(1,1), 'none')
        for mode in ['none','radar','radar-dav2']:
            target = MultiSweepTeacher(nn.Linear(1,1), mode)
            target.load_state_dict(original.state_dict(), strict=True)
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))
            broken = dict(original.state_dict())
            del broken['student.weight']
            with self.assertRaises(RuntimeError):
                target.load_state_dict(broken, strict=True)

    def test_relative_teacher_resolution_is_explicit_and_teacher_is_frozen(self):
        class Teacher(nn.Module):
            def forward(self, inputs):
                self.input_shape = tuple(inputs.shape)
                return inputs[:,0]
        teacher = Teacher()
        images = torch.ones(1,2,3,8,12,requires_grad=True)
        result = relative_depth_target(teacher, images, 'cpu', (28,42))
        self.assertEqual(teacher.input_shape, (2,3,28,42))
        self.assertEqual(tuple(result.shape), (2,28,42))
        self.assertFalse(result.requires_grad)


if __name__ == '__main__':
    unittest.main()
