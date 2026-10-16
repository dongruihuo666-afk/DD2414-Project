"""CPU contracts for opt-in depth, double distillation and unattended queue."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from baseline_followup import state_digest
from multisweep_teacher import (DepthAwareVox, Downstream, MultiSweepTeacher,
    configure, depth_gate, image_cosine_loss, object_targets, pool_objects,
    projected_radar_depth, relative_depth_loss, sparse_depth_loss, recent_radar_target_batch,
    prepare_bevcar_inputs)
from run_multisweep_followup import experiment_stages, should_stop_worker


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,128,1),nn.BatchNorm2d(128))
        self.bevcar_encoder = nn.Conv2d(128,128,1)
        self.bev_compressor = nn.Conv2d(128,128,1)
        self.decoder = nn.Conv2d(128,1,1)

    def forward(self,rgb_camXs,**kwargs):
        images = self.encoder(rgb_camXs)
        return self.bev_compressor(self.bevcar_encoder(images)),None,images


class TeacherTest(unittest.TestCase):
    def test_double_distillation_reaches_camera_and_bev_head(self):
        model = MultiSweepTeacher(Student())
        configure(model,'pretrain')
        shared,image,_ = model.features((torch.randn(2,3,4,5),None,None,None,None))
        loss = image_cosine_loss(model.image_head(image),torch.randn(1,2,384,4,5))
        loss = loss + model.bev_head(shared).square().mean()
        loss.backward()
        for p in [model.student.encoder[0].weight,model.image_head.weight,model.bev_head.weight]:
            self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.student.decoder.parameters()))

    def test_frozen_linear_head_does_not_update_bn_or_parameters(self):
        backbone = MultiSweepTeacher(Student())
        model = Downstream(backbone,'frozen-probe')
        before = state_digest(backbone)
        model.train()
        model((torch.randn(2,3,4,5),None,None,None,None)).mean().backward()
        self.assertEqual(before,state_digest(backbone))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertIsNotNone(model.probe.weight.grad)

    def test_unfrozen_same_head_reaches_camera(self):
        model = Downstream(MultiSweepTeacher(Student()),'unfrozen-probe')
        model.train()
        model((torch.randn(2,3,4,5),None,None,None,None)).square().mean().backward()
        self.assertIsNotNone(model.backbone.student.encoder[0].weight.grad)
        self.assertIsNone(model.backbone.bev_head.weight.grad)

    def test_depth_gate_rejects_wrong_depth_and_has_gradient(self):
        log_depth = torch.tensor([math.log(10.)],requires_grad=True)
        near = depth_gate(log_depth,torch.tensor([10.]))
        far = depth_gate(log_depth,torch.tensor([50.]))
        self.assertGreater(float(near),float(far))
        far.sum().backward()
        self.assertGreater(float(log_depth.grad.abs()),0)

    def test_depth_aware_vox_changes_lifted_features_and_reaches_head(self):
        class Vox:
            def unproject_image_to_mem(self,features,*args,**kwargs):
                return features[:,:,None]
        head = nn.Conv2d(2,1,1)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        features = torch.ones(1,2,2,2,requires_grad=True)
        xyz = torch.tensor([[[0.,0.,4.],[0.,0.,4.],[0.,0.,4.],[0.,0.,4.]]])
        proxy = DepthAwareVox(Vox(),head)
        result = proxy.unproject_image_to_mem(features,None,torch.eye(4)[None],1,2,2,xyz_camA=xyz)
        self.assertTrue(torch.all(result < 1))
        result.sum().backward()
        self.assertGreater(float(head.bias.grad.abs()),0)

    def test_sparse_depth_exact_and_empty(self):
        depth = torch.full((1,1,3,4),math.log(10.),requires_grad=True)
        anchors = [(torch.tensor([[1.,1.]]),torch.tensor([10.]))]
        loss,info = sparse_depth_loss(depth,anchors)
        self.assertAlmostEqual(float(loss),0.,places=6)
        self.assertLess(info['depth_mae_m'],1e-5)
        zero,info = sparse_depth_loss(depth,[(torch.empty(0,2),torch.empty(0))])
        zero.backward()
        self.assertEqual(info['depth_anchors'],0)
        self.assertIsNotNone(depth.grad)

    def test_radar_depth_projection_ignores_old_and_padded_returns(self):
        batch = [None]*18
        batch[0] = torch.zeros(1,1,1,3,10,10)
        batch[1] = torch.eye(3)[None,None,None]
        batch[2] = torch.zeros(1,1,1,3)
        batch[3] = torch.eye(4)[None,None,None]
        radar = torch.zeros(1,1,19,3)
        radar[0,0,:3,0] = torch.tensor([2.,2.,2.])
        radar[0,0,:3,1] = torch.tensor([3.,3.,3.])
        radar[0,0,18,1] = .5
        batch[16] = radar
        anchors = projected_radar_depth(batch,10,10,'cpu')
        self.assertEqual(len(anchors),1)
        self.assertEqual(anchors[0][1].tolist(),[2.])
        self.assertEqual(anchors[0][0].tolist(),[[1.,1.]])

    def test_relative_teacher_not_used_as_metric_depth(self):
        prediction = torch.log(torch.tensor([[[[1.,2.],[4.,8.]]]])).requires_grad_()
        target = torch.tensor([[[8.,4.],[2.,1.]]])
        a = relative_depth_loss(prediction,target)
        b = relative_depth_loss(prediction,target*7+19)
        self.assertTrue(torch.allclose(a,b,atol=1e-5))
        a.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_dense_ten_sweep_input_does_not_hit_old_3500_guard(self):
        batch = [None]*18
        batch[0] = torch.zeros(1,1,1,3,8,8)
        batch[1] = torch.eye(3)[None,None,None]
        batch[2] = torch.zeros(1,1,1,3)
        batch[3] = torch.eye(4)[None,None,None]
        radar = torch.zeros(1,1,19,4000)
        index = torch.arange(4000)
        # Cell centers avoid banker-rounding ties at exact cell boundaries.
        radar[0,0,0] = (index%100)*.5-24.75
        radar[0,0,2] = (index//100)*.5+1.25
        batch[16] = radar
        inputs = prepare_bevcar_inputs(batch,'cpu')
        self.assertEqual(inputs[-1][-1].tolist(),[4000])

    def test_target_support_does_not_change_with_more_old_sweeps(self):
        batch = [torch.zeros(1)]*18
        radar = torch.ones(1,1,19,2)
        radar[:,:,18,0] = 0.
        radar[:,:,18,1] = .5
        batch[16] = radar
        output = recent_radar_target_batch(batch)
        self.assertTrue(torch.equal(output[16][...,0],radar[...,0]))
        self.assertEqual(float(output[16][...,1].sum()),0.)
        self.assertEqual(float(radar[...,1].sum()),18.5)

    def test_object_pool_samples_correct_spatial_locations(self):
        features = torch.arange(16.).reshape(1,1,4,4)
        record = dict(grid=torch.tensor([[1.,1.],[-1.,-1.]]))
        self.assertEqual(float(pool_objects(features,[record])),7.5)
        self.assertEqual(pool_objects(features,[]).shape,(0,1))

    def test_gt_velocity_uses_annotation_rotation_not_radar(self):
        identity = [1.,0.,0.,0.]
        data = {('sample','s'):dict(data={'LIDAR_TOP':'l','CAM_FRONT':'c'},anns=['a']),
            ('sample_data','l'):dict(ego_pose_token='e'),
            ('ego_pose','e'):dict(translation=[0,0,0],rotation=identity),
            ('sample_data','c'):dict(calibrated_sensor_token='cs'),
            ('calibrated_sensor','cs'):dict(translation=[0,0,0],rotation=identity),
            ('sample_annotation','a'):dict(category_name='vehicle.car',visibility_token='4',
                translation=[1,0,10],rotation=identity,size=[2,4,2])}
        nusc = SimpleNamespace(get=lambda table,token:data[table,token],
                               box_velocity=lambda token:np.array([3.,0.,4.]))
        vox = SimpleNamespace(Ref2Mem=lambda value,*a,**kw:value)
        records = object_targets(SimpleNamespace(nusc=nusc),'s',vox,'cpu')
        self.assertEqual(records[0]['velocity'].tolist(),[3.,4.])
        from pyquaternion import Quaternion
        data['calibrated_sensor','cs']['rotation'] = Quaternion(axis=[0,1,0],angle=math.pi/2).elements
        records = object_targets(SimpleNamespace(nusc=nusc),'s',vox,'cpu')
        self.assertTrue(torch.allclose(records[0]['velocity'],torch.tensor([-4.,3.])))
        nusc.box_velocity = lambda token:np.array([float('nan'),0.,0.])
        self.assertEqual(object_targets(SimpleNamespace(nusc=nusc),'s',vox,'cpu'),[])

    def test_no_time_or_stale_duration_can_kill_a_worker(self):
        self.assertFalse(should_stop_worker(False,1e12,1e12))
        self.assertTrue(should_stop_worker(True,1.,0.))

    def test_queue_has_full_matched_multi_sweep_and_all_diagnostics(self):
        args = SimpleNamespace(data_root=Path('/data'),bevcar_source=Path('/source'),
            study_dir=Path('/output'),batch_size=3,num_workers=8,
            pretrain_epochs=1,downstream_epochs=1)
        stages = experiment_stages(args)
        pretraining = [t for t in stages if '--mode' in t['command'] and
                       t['command'][t['command'].index('--mode')+1] == 'pretrain']
        self.assertEqual(len(pretraining),4)
        for stage in stages:
            command = stage['command']
            self.assertNotIn('--train-samples',command)
            self.assertNotIn('--stop-after-updates',command)
        for stage in pretraining:
            self.assertIn(stage['command'][stage['command'].index('--nsweeps')+1],['5','10'])
        for mode in ['frozen-probe','unfrozen-probe','finetune','object-motion']:
            self.assertEqual(sum(t['command'][t['command'].index('--mode')+1] == mode
                for t in stages if '--mode' in t['command']),4)


if __name__ == '__main__':
    unittest.main()
