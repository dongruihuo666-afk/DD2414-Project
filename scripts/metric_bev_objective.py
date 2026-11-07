"""Two DINO terms; optional metric pseudo-depth loss after camera projection."""
import torch
from torch import nn

from fullsize_training import (extract_teacher_features_batched,
    build_targets_from_features_batched, semantic_loss_batched)
from multisweep_teacher import (image_cosine_loss, recent_radar_target_batch,
                               prepare_bevcar_inputs)
from metric_depth_bev import (metric_teacher_depth, project_teacher_depth,
                              projected_depth_loss)


class MetricPretrainingForward(nn.Module):
    def __init__(self, backbone, depth_supervision):
        super().__init__()
        self.backbone, self.depth_supervision = backbone, depth_supervision

    def forward(self, inputs):
        shared, image, _ = self.backbone.features(inputs, retain_geometry=self.depth_supervision)
        outputs = dict(image=self.backbone.image_head(image), bev=self.backbone.bev_head(shared))
        if self.depth_supervision:
            outputs.update(self.backbone.geometry)
        return outputs


def metric_pretraining_loss(batch, forward, teacher, depth_teacher, args, device):
    with torch.no_grad():
        features = extract_teacher_features_batched(teacher, batch[0][:, 0], device,
                                                    resolution=args.teacher_resolution)
        bev, mask = build_targets_from_features_batched(recent_radar_target_batch(batch), features, device)
    # Teacher extraction uses inference tensors in existing utilities; clone
    # targets before autograd needs to save them for the student's backward.
    features, bev, mask = features.clone(), bev.clone(), mask.clone()
    # Frozen teacher inference must finish before student activations are live.
    # The teacher is eval-only; this changes memory lifetime, not its targets.
    depth = (metric_teacher_depth(depth_teacher, batch[0][:, 0], device,
             args.depth_teacher_resolution) if depth_teacher is not None else None)
    outputs = forward(prepare_bevcar_inputs(batch, device))  # Full velocity; never zeroed.
    parts = dict(image=image_cosine_loss(outputs['image'], features),
                 bev=semantic_loss_batched(outputs['bev'], bev, mask))
    loss = args.image_weight * parts['image'] + args.bev_weight * parts['bev']
    if depth_teacher is not None:
        q = project_teacher_depth(depth, outputs['grid'], outputs['valid'],
                                  outputs['feature_size'], args.metric_depth_sigma)
        parts['metric_depth_bev'] = projected_depth_loss(outputs['log_probability_bev'], q)
        loss = loss + args.metric_depth_weight * parts['metric_depth_bev']
    return loss, parts, dict(bev_mask_cells=int((mask > 0).sum()))
