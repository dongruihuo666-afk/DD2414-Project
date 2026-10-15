#!/usr/bin/env python3
"""CPU tests for the hybrid-teacher losses and official metric boundary."""

import unittest

import torch
import torch.nn as nn

import utils.basic
from hybrid_teacher_full import (
    configure_linear_probe,
    image_distillation_loss,
    module_state_sha256,
    simplebev_downstream_loss,
    simplebev_per_frame_iou,
    simplebev_segmentation_loss,
)
from train_nuscenes import balanced_mse_loss


class HybridTeacherProtocolTest(unittest.TestCase):
    def test_image_distillation_is_zero_for_matching_directions(self):
        teacher = torch.randn(2, 3, 4, 2, 2)
        prediction = teacher.reshape(6, 4, 2, 2).clone()
        loss = image_distillation_loss(prediction, teacher)
        self.assertLess(float(loss), 1e-6)

    def test_image_distillation_penalizes_opposite_directions(self):
        teacher = torch.randn(1, 2, 4, 2, 2)
        prediction = -teacher.reshape(2, 4, 2, 2)
        loss = image_distillation_loss(prediction, teacher)
        self.assertAlmostEqual(float(loss), 2.0, places=5)

    def test_simplebev_iou_is_mean_of_frame_values_not_global_iou(self):
        # Frame 1 has IoU 1/1. Frame 2 has IoU 1/3, so official mean is 2/3.
        prediction = torch.tensor([
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 0.0]]],
        ])
        target = torch.tensor([
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ])
        valid = torch.ones_like(target)
        logits = torch.where(prediction > 0.5, 20.0, -20.0)
        values = simplebev_per_frame_iou(logits, target, valid)
        self.assertAlmostEqual(float(values[0]), 1.0, places=3)
        self.assertAlmostEqual(float(values[1]), 1.0 / 3.0, places=3)
        self.assertAlmostEqual(float(values.mean()), 2.0 / 3.0, places=3)
        # Global intersection/union would be 2/4 = 0.5 and is intentionally
        # different from the official Simple-BEV metric.
        self.assertNotAlmostEqual(float(values.mean()), 0.5, places=2)

    def test_invalid_cells_do_not_affect_iou(self):
        logits = torch.tensor([[[[20.0, 20.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.tensor([[[[1.0, 0.0]]]])
        self.assertAlmostEqual(
            float(simplebev_per_frame_iou(logits, target, valid)[0]),
            1.0, places=3,
        )

    def test_downstream_loss_matches_simplebev_components(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.ce_weight = nn.Parameter(torch.tensor(0.2))
                self.center_weight = nn.Parameter(torch.tensor(-0.1))
                self.offset_weight = nn.Parameter(torch.tensor(0.3))

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.student = Student()

        model = Wrapper()
        outputs = {
            'segmentation': torch.tensor([[[[0.3, -0.2]]]]),
            'center': torch.tensor([[[[0.5, 0.2]]]]),
            'offset': torch.tensor([[[[0.4, -0.1]], [[0.2, 0.3]]]]),
        }
        labels = {
            'segmentation': torch.tensor([[[[1.0, 0.0]]]]),
            'valid': torch.ones(1, 1, 1, 2),
            'center': torch.tensor([[[[0.8, 0.0]]]]),
            'offset': torch.zeros(1, 2, 1, 2),
        }
        segmentation_loss = simplebev_segmentation_loss(torch.device('cpu'))
        total, components = simplebev_downstream_loss(
            model, outputs, labels, segmentation_loss,
        )
        raw_ce = segmentation_loss(
            outputs['segmentation'], labels['segmentation'], labels['valid'],
        )
        raw_center = balanced_mse_loss(outputs['center'], labels['center'])
        offset_error = torch.abs(
            outputs['offset'] - labels['offset']
        ).sum(dim=1, keepdim=True)
        raw_offset = utils.basic.reduce_masked_mean(
            offset_error, labels['segmentation'] * labels['valid'],
        )
        expected_ce = 10.0 * raw_ce / torch.exp(model.student.ce_weight)
        expected_center = raw_center / (
            2.0 * torch.exp(model.student.center_weight)
        )
        expected_offset = raw_offset / (
            2.0 * torch.exp(model.student.offset_weight)
        )
        expected_uncertainty = 0.5 * (
            model.student.ce_weight + model.student.center_weight
            + model.student.offset_weight
        )
        expected = (
            expected_ce + expected_center + expected_offset
            + expected_uncertainty
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.allclose(
            components['segmentation_ce_loss'], expected_ce,
        ))

    def test_linear_probe_freezes_complete_backbone(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim = 3
                self.weight = nn.Parameter(torch.ones(()))

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.student = Student()
                self.extra = nn.Parameter(torch.ones(()))

            def shared_bev(self, features):
                return features * self.student.weight * self.extra

        backbone = Backbone()
        probe, trainable = configure_linear_probe(backbone)
        self.assertTrue(all(not value.requires_grad
                            for value in backbone.parameters()))
        self.assertEqual(trainable, list(probe.probe.parameters()))
        result = probe((torch.ones(2, 3, 4, 4),))
        result.mean().backward()
        self.assertTrue(all(value.grad is None for value in backbone.parameters()))
        self.assertTrue(all(value.grad is not None for value in trainable))
        initial_hash = module_state_sha256(backbone)
        with torch.no_grad():
            backbone.extra.add_(1.0)
        self.assertNotEqual(initial_hash, module_state_sha256(backbone))


if __name__ == '__main__':
    unittest.main()
