"""CPU geometry/gradient/admission tests; not a CUDA throughput or quality claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import metric_depth_bev as metric
import metric_bev_objective as objective
from metric_bev_profile import validate_metric_args
from distributed_training import AcknowledgedDistributedSampler, validation_shard
from train_multisweep_teacher import parse_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cluster_rpl'))
from random_depth_study import load_config, treatment_plan


def ray_points(depths):
    # Project to the same interior image pixel (1,1) at all metric depths.
    return torch.tensor([[[d, d, d] for d in depths]], dtype=torch.float32)


class BaseVox:
    def unproject_image_to_mem(self, features, pixels, cameras, z, y, x, **kwargs):
        return features.mean((-2, -1), keepdim=True).unsqueeze(-1).expand(-1, -1, z, y, x)


class TinyMetricBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.camera = nn.Conv2d(3, 128, 1)
        self.depth_head = nn.Conv2d(128, 80, 1)
        self.image_head = nn.Conv2d(128, 4, 1)
        self.bev_head = nn.Conv2d(128, 4, 1)

    def features(self, inputs, retain_geometry=False):
        image = self.camera(inputs)
        wrapper = metric.DepthProbabilityVox(BaseVox(), self.depth_head, retain_geometry)
        identity = torch.eye(4)[None]
        volume = wrapper.unproject_image_to_mem(image, identity, identity, 3, 1, 1,
                                                xyz_camA=ray_points((10., 20., 30.)))
        self.geometry = wrapper.geometry
        return volume.mean(3), image, None


def metric_gloo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        for supervised in (False, True):
            torch.manual_seed(125)
            model = TinyMetricBackbone()
            forward = DistributedDataParallel(objective.MetricPretrainingForward(model, supervised))
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs = forward(torch.full((1, 3, 3, 3), rank+.3+step))
                loss = .2*outputs['image'].square().mean()+.8*outputs['bev'].square().mean()
                if supervised:
                    target = metric.project_teacher_depth(torch.full((1, 1, 3, 3), 20.),
                        outputs['grid'], outputs['valid'], outputs['feature_size'])
                    loss = loss + .05*metric.projected_depth_loss(outputs['log_probability_bev'], target)
                loss.backward()
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    assert parameter.grad.abs().sum() > 0, name
                    gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                    dist.all_gather(gathered, parameter.grad)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
                optimizer.step()
    finally:
        dist.destroy_process_group()


class MetricDepthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(125)
        self.features = torch.randn(1, 128, 3, 3, requires_grad=True)
        self.identity = torch.eye(4)[None]

    def grid(self, depths=(10., 20., 30.)):
        return metric.ray_grid(None, self.features, self.identity, self.identity,
                               len(depths), 1, 1, ray_points(depths))

    def test_depth_bin_centers_sample_exact_metric_values(self):
        grid, valid = self.grid((1., 10., 20., 80.))
        bins = torch.arange(1., 81.)[None, None, :, None, None].expand(1, 1, 80, 3, 3)
        actual = metric.sample_ray_volume(bins, grid).flatten()
        torch.testing.assert_close(actual, torch.tensor([1., 10., 20., 80.]), atol=1e-5, rtol=0)
        self.assertTrue(valid.all())

    def test_teacher_localizes_twenty_meters_not_entire_ray(self):
        grid, valid = self.grid()
        depth = torch.full((1, 1, 3, 3), 20., requires_grad=True)
        target = metric.project_teacher_depth(depth, grid, valid, (3, 3))
        self.assertGreater(float(target.flatten()[1]), 1000*float(target.flatten()[0]))
        self.assertFalse(target.requires_grad)

    def test_camera_extrinsics_change_metric_depth_consistently_with_intrinsics(self):
        camera = self.identity.clone()
        camera[:, 2, 3] = -5.
        intrinsics = self.identity.clone()
        intrinsics[:, 0, 2] = 1.
        intrinsics[:, 1, 2] = 1.
        grid, valid = metric.ray_grid(None, self.features, intrinsics@camera, camera,
                                     1, 1, 1, torch.tensor([[[0., 0., 20.]]]))
        bins = torch.arange(1., 81.)[None, None, :, None, None].expand(1, 1, 80, 3, 3)
        actual = metric.sample_ray_volume(bins, grid)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(actual), 15., places=4)

    def test_wrong_distance_has_larger_loss_and_gradient_moves_mass_to_twenty(self):
        grid, valid = self.grid()
        target = metric.project_teacher_depth(torch.full((1, 1, 3, 3), 20.), grid, valid, (3, 3))
        logits = torch.zeros(1, 80, 3, 3, requires_grad=True)
        def loss_at(values):
            return metric.projected_depth_loss(metric.sample_ray_volume(
                F.log_softmax(values, 1)[:, None], grid), target)
        value = loss_at(logits)
        value.backward()
        self.assertLess(float(logits.grad[:, 19].sum()), 0.)
        self.assertGreater(float(logits.grad[:, 9].sum()), 0.)
        correct, wrong = logits.detach().clone(), logits.detach().clone()
        correct[:, 19] = 5
        wrong[:, 9] = 5
        self.assertLess(float(loss_at(correct)), float(loss_at(wrong)))
        self.assertTrue(torch.isfinite(value))

    def test_projection_routes_features_and_backpropagates_into_camera(self):
        head = nn.Conv2d(128, 80, 1)
        wrapper = metric.DepthProbabilityVox(BaseVox(), head, True)
        volume = wrapper.unproject_image_to_mem(self.features, self.identity,
            self.identity, 3, 1, 1, xyz_camA=ray_points((10., 20., 30.)))
        geometry = wrapper.geometry
        target = metric.project_teacher_depth(torch.full((1, 1, 3, 3), 20.),
            geometry['grid'], geometry['valid'], (3, 3))
        loss = metric.projected_depth_loss(geometry['log_probability_bev'], target)
        loss.backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.features.grad.abs().sum()), 0.)
        self.assertEqual(tuple(volume.shape), (1, 128, 3, 1, 1))
        # Geometry weights actually affect lifting, not merely a detached head.
        self.assertFalse(torch.equal(volume[:, :, 0], volume[:, :, 1]))

    def test_invalid_teacher_and_out_of_range_voxels_have_finite_zero_loss(self):
        grid, valid = self.grid((-.1, .5, 100.))
        self.assertFalse(valid.any())
        target = metric.project_teacher_depth(torch.full((1, 1, 3, 3), float('nan')),
                                               grid, valid, (3, 3))
        values = torch.randn_like(target, requires_grad=True)
        loss = metric.projected_depth_loss(values, target)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(values.grad, torch.zeros_like(values)))

    def test_control_and_treatment_have_identical_initial_student_state(self):
        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
        def factory(device, seed, source, mode):
            torch.manual_seed(seed)
            return metric.MultiSweepTeacher(TinyStudent())
        with patch.object(metric, 'build_model', factory):
            a = metric.build_metric_model('cpu', 125, None)
            b = metric.build_metric_model('cpu', 125, None)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            b.load_state_dict(metric.MultiSweepTeacher(TinyStudent()).state_dict(), strict=True)

    def test_only_dino_terms_in_control_and_optional_depth_in_treatment(self):
        class TinyForward:
            def __call__(self, inputs):
                value = torch.ones(1, 2, 2, 2, requires_grad=True)
                return dict(image=value, bev=value, grid=None, valid=None,
                            feature_size=(2, 2), log_probability_bev=value)
        args = type('Args', (), dict(teacher_resolution=[448, 798], depth_teacher_resolution=[448, 798],
                                    image_weight=.2, bev_weight=.8, metric_depth_sigma=2., metric_depth_weight=.05))()
        mse = lambda a, b, *rest: (a-b).square().mean()
        target = torch.zeros(1, 2, 2, 2)
        dependencies = dict(extract_teacher_features_batched=lambda *a, **k: target,
            build_targets_from_features_batched=lambda *a, **k: (target, torch.ones_like(target)),
            recent_radar_target_batch=lambda a: a, prepare_bevcar_inputs=lambda *a: (),
            image_cosine_loss=mse, semantic_loss_batched=mse,
            metric_teacher_depth=lambda *a: target, project_teacher_depth=lambda *a: target,
            projected_depth_loss=lambda *a: torch.tensor(2.))
        with patch.multiple(objective, **dependencies):
            for teacher, expected in [(None, 1.), (object(), 1.1)]:
                loss, parts, _ = objective.metric_pretraining_loss([torch.zeros(1, 1)], TinyForward(),
                                                                  object(), teacher, args, 'cpu')
                self.assertAlmostEqual(float(loss), expected, places=6)
                self.assertEqual(set(parts), {'image', 'bev'} | ({'metric_depth_bev'} if teacher else set()))

    def test_new_plans_parse_and_require_random_fifty_epoch_protocol(self):
        config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
        for treatment in metric.TREATMENTS:
            task = treatment_plan(config, treatment)['pretrain']
            argv = task['argv']
            with patch.object(sys, 'argv', ['trainer', *argv[argv.index(str(ROOT/'scripts/train_multisweep_teacher.py'))+1:]]):
                args = parse_args()
            environment = dict(WORLD_SIZE='5', LOCAL_WORLD_SIZE='5', RANK='0', LOCAL_RANK='0')
            validate_metric_args(args, environment)
            self.assertEqual((task['epochs'], task['updates_per_epoch'], task['total_training_sample_visits']),
                              (50, 5626, 1406500))
            for key, changed in [('official_camera_checkpoint', Path('/forbidden')), ('epochs', 5),
                                 ('motion_weight', .5), ('velocity_mode', 'zero')]:
                bad = copy.deepcopy(args)
                setattr(bad, key, changed)
                with self.assertRaises(ValueError):
                    validate_metric_args(bad, environment)

    def test_supported_full_epoch_shards_and_uneven_validation_cover_once(self):
        data = range(28130)
        for world in (2, 5, 10):
            shards = [list(AcknowledgedDistributedSampler(data, 125, rank, world)) for rank in range(world)]
            combined = [i for shard in shards for i in shard]
            self.assertEqual(sorted(combined), list(data))
            self.assertEqual(len(set(combined)), len(data))
            validation = [i for rank in range(world) for i in validation_shard(list(range(6019)), rank, world)]
            self.assertEqual(sorted(validation), list(range(6019)))
        for world in (4, 8):
            with self.assertRaises(ValueError):
                AcknowledgedDistributedSampler(data, 125, 0, world)

    def test_real_two_process_gradient_sync_for_both_new_forward_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = 'file://' + str(Path(temporary)/'rendezvous')
            mp.spawn(metric_gloo_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
