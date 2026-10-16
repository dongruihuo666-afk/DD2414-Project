"""Meaningful CPU checks for the legacy downstream protocols."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from baseline_followup import (LegacyDownstream, official_counts, state_digest,
                               supervised_loss)
from train_baseline_downstream import parameter_groups
from train_fullsize_linear_probe import metrics_from_counts
from train_nuscenes import SimpleLoss, balanced_mse_loss
from fullsize_training import AcknowledgedShuffleSampler
from run_baseline_followup_day import choose_epochs
from report_baseline_followup import paired_scene_interval
from training_checkpoint import load_training_checkpoint, save_training_checkpoint
import utils.basic


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 3
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.bevcar_encoder = nn.Conv2d(3, 3, 1)
        self.bev_compressor = nn.Conv2d(3, 3, 1)
        self.decoder = nn.Conv2d(3, 3, 1)
        for name in ['ce_weight', 'center_weight', 'offset_weight']:
            setattr(self, name, nn.Parameter(torch.tensor(.1)))

    def forward(self, rgb_camXs, return_shared_bev=False, **kwargs):
        features = self.bev_compressor(self.bevcar_encoder(self.encoder(rgb_camXs)))
        return features, None, None


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = Student()
        self.semantic_head = nn.Conv2d(3, 2, 1)
        self.motion_head = nn.Conv2d(3, 2, 1)


class BaselineFollowupTest(unittest.TestCase):
    def test_scene_bootstrap_same_predictions_has_zero_difference(self):
        counts = {'scene1':[3,2,1], 'scene2':[1,2,3]}
        self.assertEqual(paired_scene_interval(counts, counts, draws=100), [0.,0.])

    def test_epoch_budget_is_selected_for_the_pair_before_training(self):
        self.assertEqual(choose_epochs(10., 20.), 3)
        self.assertEqual(choose_epochs(1., 10.), 1)

    def test_global_iou_and_official_half_probability_tie(self):
        counts = official_counts(torch.tensor([0., 9., 9., 9.]),
            torch.tensor([0., 1., 0., 0.]), torch.tensor([1., 1., 1., 0.]))
        self.assertEqual(counts, dict(tp=1, fp=1, fn=0, tn=1))
        self.assertEqual(metrics_from_counts(counts)['vehicle_iou'], .5)

    def test_loss_matches_original_simplebev_components(self):
        student = Student()
        seg = torch.tensor([[[[.4, -.2]]]])
        center = torch.tensor([[[[.2, .6]]]])
        offset = torch.ones(1, 2, 1, 2)
        labels = dict(segmentation=torch.tensor([[[[1., 0.]]]]),
            valid=torch.ones_like(seg), center=torch.tensor([[[[1., 0.]]]]),
            offset=torch.zeros_like(offset))
        criterion = SimpleLoss(2.13)
        total, parts, prediction = supervised_loss(student,
            (None, None, seg, center, offset), labels, criterion)
        expected = (10*criterion(seg, labels['segmentation'], labels['valid'])
            /student.ce_weight.exp()
            + balanced_mse_loss(center, labels['center'])/(2*student.center_weight.exp())
            + utils.basic.reduce_masked_mean(offset.abs().sum(1, keepdim=True),
                labels['segmentation']*labels['valid'])/(2*student.offset_weight.exp())
            + .5*(student.ce_weight+student.center_weight+student.offset_weight))
        self.assertTrue(torch.allclose(total, expected))
        self.assertIs(prediction, seg)

    def test_frozen_probe_preserves_parameters_and_batchnorm_buffers(self):
        model = LegacyDownstream(Backbone(), 'frozen-probe')
        before = state_digest(model.backbone)
        model.train()
        inputs = (torch.randn(2, 3, 4, 4), None, None, None, None)
        with patch('baseline_followup.prepare_bevcar_inputs', return_value=inputs), \
                patch('baseline_followup.apply_radar_velocity_mode', return_value=None):
            result = model(None, 'cpu', 'full')
        result.mean().backward()
        self.assertEqual(before, state_digest(model.backbone))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.probe.parameters()))

    def test_unfrozen_same_head_reaches_encoder_and_excludes_unused_heads(self):
        model = LegacyDownstream(Backbone(), 'unfrozen-probe')
        model.train()
        inputs = (torch.randn(2, 3, 4, 4), None, None, None, None)
        with patch('baseline_followup.prepare_bevcar_inputs', return_value=inputs), \
                patch('baseline_followup.apply_radar_velocity_mode', return_value=None):
            model(None, 'cpu', 'full').square().mean().backward()
        self.assertIsNotNone(model.backbone.student.encoder[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.motion_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.student.decoder.parameters()))

    def test_parameter_groups_cover_each_trainable_parameter_once(self):
        model = LegacyDownstream(Backbone(), 'finetune')
        args = SimpleNamespace(camera_lr=1e-5, learning_rate=1e-4, probe_lr=1e-3)
        groups = parameter_groups(model, args)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_resume_restores_adam_sampler_and_next_update(self):
        torch.manual_seed(125)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        sampler = AcknowledgedShuffleSampler(range(4), 125)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        def update():
            optimizer.zero_grad()
            model(torch.randn(2, 2)).square().mean().backward()
            optimizer.step()
            sampler.acknowledge(2)
        update()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'latest.pt'
            config = {'epochs': 1}
            save_training_checkpoint(checkpoint, model=model, optimizer=optimizer,
                scaler=scaler, sampler=sampler, progress={'epoch':0, 'update':1}, run_config=config)
            update()
            expected = state_digest(model)
            load_training_checkpoint(checkpoint, model=model, optimizer=optimizer,
                scaler=scaler, sampler=sampler, expected_run_config=config, map_location='cpu')
            self.assertEqual(sampler.position, 2)
            update()
            self.assertEqual(state_digest(model), expected)


if __name__ == '__main__':
    unittest.main()
