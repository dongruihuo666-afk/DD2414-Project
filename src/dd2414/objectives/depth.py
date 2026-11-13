from dd2414.data.batch import as_batch
"""DDP counterpart of the existing single-GPU research pretraining objective.

The legacy path stays unchanged. A CPU equivalence test checks all three depth
treatments against its actual pretrain_loss closure, including student gradients.
"""
import torch
from torch import nn

from dd2414.runtime.fullsize import extract_teacher_features_batched, build_targets_from_features_batched, semantic_loss_batched, build_motion_targets_batched, motion_loss_batched
from dd2414.models.multisweep import image_cosine_loss, projected_radar_depth, relative_depth_loss, relative_depth_target, sparse_depth_loss, recent_radar_target_batch, prepare_bevcar_inputs
from dd2414.evaluation.radar_controls import apply_radar_velocity_mode


class PretrainingForward(nn.Module):
    """All trainable heads execute INSIDE DDP.forward, never via .module bypass."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs):
        shared, image, depth = self.backbone.features(inputs)
        return dict(image=self.backbone.image_head(image),
            bev=self.backbone.bev_head(shared), motion=self.backbone.motion_head(shared),
            depth=depth)


def pretraining_loss(batch, forward, teacher, depth_teacher, args, device):
    target_batch = recent_radar_target_batch(batch)
    with torch.no_grad():
        features = extract_teacher_features_batched(teacher,as_batch(batch).images[:,0],device,
                                                   resolution=args.teacher_resolution)
        bev, mask = build_targets_from_features_batched(target_batch,features,device)
    bev, mask = bev.clone(), mask.clone()
    inputs = prepare_bevcar_inputs(batch,device)
    inputs = (*inputs[:-1],apply_radar_velocity_mode(inputs[-1],args.velocity_mode))
    outputs = forward(inputs)
    parts = dict(image=image_cosine_loss(outputs['image'],features),
                 bev=semantic_loss_batched(outputs['bev'],bev,mask))
    loss = args.image_weight*parts['image']+args.bev_weight*parts['bev']
    info = dict(bev_mask_cells=int((mask>0).sum()),
        radar_points_per_sample=(as_batch(batch).radar[:,0,:3].abs().sum(1)>0).sum(1).tolist(),
        occupied_voxels_per_sample=inputs[-1][-1].tolist())
    if args.motion_weight:
        motion, coverage, _ = build_motion_targets_batched(target_batch,device)
        parts['radar_motion_auxiliary'], cells = motion_loss_batched(outputs['motion'],motion,coverage)
        info['motion_auxiliary_covered_cells'] = cells
        loss = loss+args.motion_weight*parts['radar_motion_auxiliary']
    else:
        # Keep every returned differentiable DDP output in the backward graph.
        loss = loss+outputs['motion'].sum()*0
    if outputs['depth'] is not None:
        depth = outputs['depth']
        anchors = projected_radar_depth(batch,*depth.shape[-2:],device)
        parts['radar_depth'], diagnostics = sparse_depth_loss(depth,anchors)
        info.update(diagnostics)
        loss = loss+args.depth_weight*parts['radar_depth']
        if depth_teacher is not None:
            target = relative_depth_target(depth_teacher,as_batch(batch).images[:,0],device,
                                           args.depth_teacher_resolution)
            parts['relative_depth'] = relative_depth_loss(depth,target)
            loss = loss+args.relative_depth_weight*parts['relative_depth']
    return loss, parts, info
