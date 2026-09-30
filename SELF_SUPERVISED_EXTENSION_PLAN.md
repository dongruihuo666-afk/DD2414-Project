# Self-supervised camera-radar BEV extension plan

This is the implementation plan and status. The frozen DINOv2 extraction,
geometry reuse, radar-anchored feature sampling, soft-region target generation,
feature caching, visualization, student semantic projection head, masked cosine
loss, and one- and four-sample training/checkpoint smoke tests are implemented.
The four-sample student is randomly initialized and does not load a supervised
BEV checkpoint. The radar motion branch and dataset-scale pretraining are not
implemented yet.

## 0. Implemented milestone: radar-anchored DINO target

Run:

```bash
./scripts/run_dinov2_bev_demo.sh
```

The implementation in `scripts/dinov2_bev_demo.py`:

1. loads a frozen DINOv2 ViT-S/14 teacher;
2. extracts `(6,384,14,24)` patch maps from the six cameras;
3. reuses the Simple-BEV camera transforms and intrinsics;
4. demonstrates why naive ray-wise BEV projection is depth-ambiguous;
5. uses each measured radar XYZ as a depth anchor to sample the visible-camera
   DINO features;
6. splats the sampled feature into a local Gaussian BEV region whose radius
   grows with range; and
7. caches both raw teacher maps and `(384,200,200)` soft targets without using
   any 3D box annotation.

The first mini sample has 403 valid and camera-visible radar returns, 16,064
soft-supervised BEV cells, and a measured peak allocation of 1.53 GiB. These
numbers are a pipeline check, not an accuracy result.

## 1. Shared BEV representation

Use `feat_bev` in `nets/segnet.py`, inside `Segnet.forward()`, immediately after
`self.bev_compressor(...)` and before `self.decoder(...)`. It is
`(B, 128, Z, X)` and has three useful properties: all cameras are already in a
common metric frame, radar is already fused when enabled, and it is not yet
specialized by the supervised segmentation/center/offset decoder.

`Decoder`'s `feat_e` is a reasonable downstream feature but a weaker pretraining
target because it already contains task-specific spatial processing. Keep the
existing return values and add small projection heads on `feat_bev` during a
future implementation.

## 2. DINOv2 teacher extraction

The cleanest place is the training wrapper, next to `train_nuscenes.run_model()`
after `rgb_camXs` is formed but before calling `Segnet`. Run a frozen DINOv2
teacher under `torch.no_grad()` on the packed `(B*S,3,H,W)` images. This keeps
teacher normalization, resolution, and checkpoint management outside the
student `Segnet` and avoids contaminating the deployable model.

Extract patch-token maps from a small DINOv2 backbone first (ViT-S/14). Preserve
the mapping from patch centers to resized image coordinates. Do not use the
current ImageNet-normalized student tensor blindly: apply DINOv2's documented
normalization to a separate view of the same augmentation.

## 3. Aligning teacher features to BEV

Convert teacher tokens to dense per-camera feature maps, scale `pix_T_cams` to
that feature resolution with `utils.geom.scale_intrinsics()`, and reuse
`Vox_util.unproject_image_to_mem()` as `Segnet.forward()` already does for
student camera features. Then:

1. obtain `(B,S,C_teacher,Z,Y,X)` teacher volumes;
2. mask voxels that project outside each camera;
3. combine cameras with confidence-weighted or masked averaging;
4. collapse the vertical dimension to `(B,C_teacher,Z,X)` with a small frozen
   or learned projector; and
5. match a projection of student `feat_bev` using cosine loss or normalized
   smooth L1 only where teacher coverage is valid.

This is still geometry-based distillation. It does not require vehicle boxes.
Depth ambiguity was observed directly as radial artifacts in the prototype, so
the implemented target uses radar-supported soft regions instead of pretending
every point along a camera ray is equally reliable.

## 4. Radar Doppler motion target

`nuscenesdataset.get_radar_data()` preserves native radar fields and appends
time lag. In the nuScenes 18-field radar layout, compensated planar velocity is
in native fields `vx_comp` and `vy_comp`; after separating XYZ in
`train_nuscenes.run_model()`, these are part of `meta_rad`.

Attach a small BEV motion head to `feat_bev`, predicting either 2D compensated
velocity plus confidence or a static/dynamic logit and velocity residual. Build
targets by splatting radar points into the same BEV cells and use a robust L1
or Huber loss only on radar-supported cells. Rotate velocity vectors into the
reference-camera/ego frame consistently with radar XYZ. Reject invalid/ambiguous
points using radar validity fields and downweight distant, low-RCS returns.

The motion loss should not directly constrain segmentation logits: motion is a
property of the shared representation, while the segmentation head is a
downstream task that may include parked vehicles.

## 5. Joint objective

A practical first objective is:

```text
L = lambda_sem * L_DINO_BEV
  + lambda_motion * L_Doppler
  + lambda_corr * L_soft_region
  + lambda_reg * L_feature_regularization
```

Normalize each loss by its number of valid cells. Start semantic distillation
first, linearly warm up the motion weight, and log gradient norms into the
shared compressor. Fixed weights are easier to debug than immediately reusing
the baseline's learned uncertainty scalars. If the two tasks conflict, use
gradient clipping and then consider uncertainty weighting or gradient surgery.

## 6. Soft-region correspondence insertion

Insert it after both DINO teacher features and student `feat_bev` have been
mapped into `(Z,X)` coordinates, before loss reduction. For each radar return,
construct a local Gaussian/anisotropic region whose radius grows with range and
measurement uncertainty. Compare the radar-anchored student feature with a
weighted neighborhood of teacher BEV features rather than a single hard cell.

This can live in a new loss module called by the training wrapper; it should not
be embedded in `bev_compressor`, because it is a training-time correspondence
mechanism rather than an inference-time fusion operation.

## 7. Labels removable during pretraining

Self-supervised pretraining can omit all outputs produced from human 3D boxes:
`lrtlist`, `vislist`, `tidlist`, `scorelist`, `seg_bev`, `valid_bev`,
`center_bev`, and `offset_bev`. It still needs sensor data plus calibration,
timestamps, and ego poses. Those are geometric metadata, not semantic labels.

A future pretraining loader should avoid calling `get_lrtlist()`,
`get_seg_bev()`, and `get_center_and_offset_bev()` so annotation leakage and
unnecessary CPU work are both removed.

## 8. Labels still needed downstream

For linear probing or fine-tuning, retain vehicle BEV occupancy and its valid
mask. Instance center/offset labels are needed only if instance-aware outputs
remain an evaluation target. Static/dynamic evaluation needs an independent
ground-truth motion definition, ideally derived from annotated track velocity
rather than the radar targets used for pretraining. Near/far evaluation can bin
ground-truth BEV cells or box centers by metric distance and uses the same
semantic annotations as the main downstream task.

Keep pretraining scenes and evaluation labels logically separated even on mini;
mini results are pipeline checks, not statistically meaningful benchmarks.

## 9. Minimum viable prototype for 8 GiB

1. Keep batch size 1, six cameras, 112x192 student inputs, EfficientNet-B0,
   AMP, one radar sweep, and the existing 200x200 BEV grid.
2. Freeze DINOv2 ViT-S/14. Prefer offline teacher-feature caching per resized
   camera image so teacher activations do not coexist with the 3D unprojection
   graph. If online extraction is required, process cameras in chunks under
   no-grad and immediately move compact teacher maps to CPU or half precision.
3. Add only two small heads on `feat_bev`: a semantic projection head and a
   2D velocity/confidence head. Do not add multi-sweep temporal attention yet.
4. Train semantic distillation on teacher-covered cells and motion on valid
   radar cells. Add soft-region weighting only after hard-cell alignment has
   been numerically verified.
5. Validate overfit behavior on 1-4 samples, then run tens of mini iterations.
   Track valid-cell counts, loss scales, gradient norms, and peak memory.
6. Only after this works, increase to 3-5 sweeps, add temporal consistency, and
   report static/dynamic plus near/far downstream slices.

The measured baseline leaves useful headroom: the tested camera and metaradar
one-step paths used about 3.42 GiB, while the full five-step camera training
peaked at 3.58 GiB. Online DINOv2 may still dominate memory, which is why cached
teacher maps are the safest first prototype.

## 10. Next implementation order

Steps 1-3 below are complete in `scripts/semantic_distill_smoke.py`: the shared
`(B,128,200,200)` feature is exposed without invoking the supervised decoder, a
small `128 -> 384` head uses confidence-masked cosine loss, and a 20-step sample
overfit reduced loss from 0.9858 to 0.2039 while preserving finite compressor
gradients and checkpoint reload. Peak allocation was 3.54 GiB.

The original multi-sample smoke item is now complete: four separately cached
teacher targets were used for 60 alternating updates. Mean pre/post DINO loss
was 1.023/0.196 in the pre-push regression, with no supervised BEV checkpoint
and no box-derived loss.

The remaining order is:

1. Add a label-free pretraining loader and a held-out target split so evaluation
   measures more than memorization of four samples.
2. Decode compensated radar velocity into the reference frame, visualize the
   motion target, then add a motion/confidence head and robust loss.
3. Replace the current direct meta-radar voxel concatenation with a compact
   learned point/voxel radar encoder to match the proposal's VoxelNet-style
   branch.
4. Run the proposal ablations: baseline, DINO-only, radar-motion-only, and both;
   then linear probe/fine-tune and report near/far plus static/dynamic slices.
