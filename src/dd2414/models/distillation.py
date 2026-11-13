import torch
import torch.nn as nn
class SemanticDistillationModel(nn.Module):
    def __init__(self, student, teacher_dim=384, image_distill=False):
        super().__init__()
        self.student = student
        self.image_distill = image_distill
        self.semantic_head = nn.Sequential(
            nn.Conv2d(student.latent_dim, student.latent_dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(student.latent_dim),
            nn.GELU(),
            nn.Conv2d(student.latent_dim, teacher_dim, 1),
        )
        if image_distill:
            self.image_head = nn.Sequential(
                nn.Conv2d(student.feat2d_dim, student.feat2d_dim,
                          3, padding=1, bias=False),
                nn.InstanceNorm2d(student.feat2d_dim),
                nn.GELU(),
                nn.Conv2d(student.feat2d_dim, teacher_dim, 1),
            )
        else:
            self.image_head = None

    def forward(self, rgb, pixel_from_cameras, camera0_from_cameras,
                vox_util, radar_voxels):
        shared_bev, _, image_features = self.student(
            rgb_camXs=rgb,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            rad_occ_mem0=radar_voxels,
            return_shared_bev=True,
        )
        fusion_prediction = self.semantic_head(shared_bev)
        image_prediction = (
            self.image_head(image_features) if self.image_head is not None else None
        )
        return fusion_prediction, image_prediction, shared_bev, image_features
