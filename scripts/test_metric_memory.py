"""Numerical/gradient equivalence and saved-activation checks for streamed lifting."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

import metric_depth_bev as metric
import utils.basic
import utils.vox


def fixture(batch=2, views=3, channels=4, shape=(3, 1, 2)):
    torch.manual_seed(125)
    features = torch.randn(batch*views, channels, 5, 6)
    features[:, 0] = 0.  # Preserve the legacy zero-cell gradient convention.
    cameras = torch.eye(4).repeat(batch*views, 1, 1)
    cameras[:, 2, 3] = torch.arange(batch*views).float() * .5
    pixels = cameras.clone()
    count = shape[0]*shape[1]*shape[2]
    depth = torch.linspace(-2, 90, count)
    xyz = depth[None, :, None].expand(batch*views, -1, 3).clone()
    # Explicit coordinates avoid CUDA-only default grid allocation in legacy Vox.
    base = object.__new__(utils.vox.Vox_util)
    return features, pixels, cameras, xyz, base


def reference(wrapper, features, pixels, cameras, xyz, batch, views, shape):
    packed = wrapper.unproject_image_to_mem(features, pixels, cameras, *shape, xyz_camA=xyz)
    volumes = packed.reshape(batch, views, *packed.shape[1:])
    return utils.basic.reduce_masked_mean(volumes, (volumes.abs() > 0).float(), dim=1)


class MetricMemoryTests(unittest.TestCase):
    def test_streamed_outputs_geometry_and_camera_head_gradients_match_reference(self):
        for supervised in (False, True):
            batch, views, shape = 2, 3, (3, 1, 2)
            values, pixels, cameras, xyz, base = fixture(batch, views, shape=shape)
            initial = nn.Conv2d(values.shape[1], 80, 1)
            results = []
            for streamed in (False, True):
                features = values.clone().requires_grad_()
                head = copy.deepcopy(initial)
                wrapper = metric.DepthProbabilityVox(base, head, supervised)
                if streamed:
                    output = wrapper.reduce_camera_features(features, pixels, cameras, *shape,
                        batch_size=batch, camera_count=views, xyz_camA=xyz)
                else:
                    output = reference(wrapper, features, pixels, cameras, xyz, batch, views, shape)
                geometry = wrapper.geometry
                loss = output.square().sum()
                if supervised:
                    q = metric.project_teacher_depth(torch.full((batch*views, 1, 5, 6), 30.),
                        geometry['grid'], geometry['valid'], geometry['feature_size'])
                    loss = loss + .05*metric.projected_depth_loss(geometry['log_probability_bev'], q)
                loss.backward()
                results.append((output.detach(), features.grad, head.weight.grad, head.bias.grad,
                                None if geometry is None else geometry['log_probability_bev'].detach()))
            for before, after in zip(*results):
                if before is None:
                    self.assertIsNone(after)
                else:
                    torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-6)

    def test_frozen_inference_uses_same_reduction_without_checkpoint(self):
        values, pixels, cameras, xyz, base = fixture()
        head = nn.Conv2d(values.shape[1], 80, 1).requires_grad_(False)
        wrapper = metric.DepthProbabilityVox(base, head)
        with torch.no_grad(), patch.object(metric, 'checkpoint', side_effect=AssertionError('recompute')):
            actual = wrapper.reduce_camera_features(values, pixels, cameras, 3, 1, 2,
                batch_size=2, camera_count=3, xyz_camA=xyz)
            expected = reference(wrapper, values, pixels, cameras, xyz, 2, 3, (3, 1, 2))
        torch.testing.assert_close(actual, expected)

    def test_saved_activations_shrink_and_projection_is_recomputed_one_view_at_a_time(self):
        batch, views, shape = 1, 6, (40, 4, 30)
        values, pixels, cameras, xyz, base = fixture(batch, views, channels=16, shape=shape)
        initial = nn.Conv2d(16, 80, 1)
        saved, calls = [], []
        for streamed in (False, True):
            sizes, camera_batches = [], []
            features = values.clone().requires_grad_()
            head = copy.deepcopy(initial)
            wrapper = metric.DepthProbabilityVox(base, head, True)
            original = base.unproject_image_to_mem
            def record(*args, **kwargs):
                camera_batches.append(len(args[0]))
                return original(*args, **kwargs)
            def pack(tensor):
                sizes.append(tensor.numel()*tensor.element_size())
                return tensor
            with patch.object(base, 'unproject_image_to_mem', record), \
                    torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                if streamed:
                    output = wrapper.reduce_camera_features(features, pixels, cameras, *shape,
                        batch_size=batch, camera_count=views, xyz_camA=xyz)
                else:
                    output = reference(wrapper, features, pixels, cameras, xyz, batch, views, shape)
                loss = output.square().mean()+wrapper.geometry['log_probability_bev'].mean()
                loss.backward()
            saved.append(sum(sizes))
            calls.append(camera_batches)
        self.assertLess(saved[1], saved[0]/2)
        self.assertEqual(calls[0], [views])
        self.assertEqual(calls[1], [1]*(2*views))


if __name__ == '__main__':
    unittest.main()
