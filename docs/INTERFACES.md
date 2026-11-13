# Python integration contract

This is a Python/Tensor boundary, not a network service. The authoritative
implementation is `src/dd2414`; legacy `nets`, `utils` and `nuscenesdataset.py`
remain installed for compatibility. No model is created on package import.

## Batch view

```python
from dd2414.data.batch import as_batch
view = as_batch(legacy_batch)  # zero copy; no transforms or implicit validation
view.validate()              # optional full-loader shape check at a boundary
images, radar = view.images, view.radar
```

`NuScenesData.__getitem__` returns the following 18-field sequence after
collation. The view retains the sequence and exposes exactly these names:

| Position | Field | Collated shape / meaning |
| --- | --- | --- |
| 0 | images | float32 `[B,T,S,3,H,W]`, RGB in `[0,1]` |
| 1 | camera_rotations | float32 `[B,T,S,3,3]`, camera to ego rotation |
| 2 | camera_translations | float32 `[B,T,S,3]`, camera origin in ego, metres |
| 3 | intrinsics | float32 `[B,T,S,4,4]`, resized/cropped homogeneous pixel calibration |
| 4–7 | lidar_current, lidar_current_extra, lidar, lidar_extra | padded channel-first point tensors; original loader semantics |
| 8–11 | boxes, visibility, track_ids, scores | original padded box/identity tensors; not student predictions |
| 12 | vehicle_mask | float32 `[B,T,1,Z,X]`, supervised binary vehicle GT |
| 13 | valid_mask | float32 `[B,T,1,Z,X]`, supervised evaluation/loss validity |
| 14 | centers | float32 `[B,T,1,Z,X]`, supervised center target |
| 15 | offsets | float32 `[B,T,2,Z,X]`, supervised grid-cell offsets |
| 16 | radar | float32 `[B,T,19,R]`, padded radar returns, `R=700*nsweeps` |
| 17 | ego_pose | original homogeneous ego pose sequence |

Research training selects time index zero, normally `T=1`. Loader outputs are
CPU tensors. Input preparation explicitly moves/casts relevant fields to the
selected device. The view never does so. Mixed dtype intermediate tensors are
controlled by the existing autocast path; source coordinates remain float32.

## Images and calibration

The deterministic research camera list starts as FRONT_LEFT, FRONT,
FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT (each prefixed `CAM_`). With
`refcam_id=1` and camera shuffling disabled, the loader swaps slots 0 and 1:
FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT. This is a swap,
not a rotation of the list. Official training may shuffle/select reference
cameras; always pass the matching calibration from that batch.

The student receives `images[:,0].float() - 0.5`. DINO instead receives resized
RGB normalized by ImageNet mean `(0.485,0.456,0.406)` and standard deviation
`(0.229,0.224,0.225)`; teacher interpolation/patch grid stay profile-specific.
Do not apply the student's centering twice or feed it DINO normalization.

In names `A_from_B`, the transform maps column coordinates in B into A.
`ego_from_cameras` comes from camera rotations/translations; its inverse maps
ego points into each camera. `camera0_from_cameras` is the reference-camera
relative transform. Camera-0 axes are x right, y down, z forward. Pixel matrices
already include image resize/crop adjustments. Metres and pixels are distinct.

## Radar and geometry

The 19 fields are nuScenes' 18 raw radar fields plus age in seconds:
`x,y,z,dyn_prop,id,rcs,vx,vy,vx_comp,vy_comp,is_quality_valid,ambig_state,
x_rms,y_rms,invalid_state,pdh0,vx_rms,vy_rms,age`.
XYZ is in metres, RCS in the original radar units, velocities in metres/second.
Zero-padded points are excluded by the existing XYZ nonzero test; quality,
finiteness and age filters depend on the calling objective. No new filtering
is introduced by this interface. The research VoxelNet path has the unchanged
7,000-occupied-voxel bound for ten sweeps.

Five physical radars are aggregated, each up to `nsweeps` current/past records.
`nsweeps=10` means ten temporal sweeps per sensor, not ten physical radars.
XYZ is transformed using the past sensor/ego pose and current RADAR_FRONT ego
reference. Age is current reference timestamp minus return timestamp in seconds.
This compensates ego poses; moving-object trajectories are not deskewed to GT.
The opt-in `rotate_radar_velocity=True` rotates both raw and ego-compensated
planar pairs through the same rotation. Translation never changes velocity.
Research input preparation rotates into camera-0 and stores camera `(vx,vz)`
in each pair. Motion targets use compensated fields 8:10; older records without
the rotation flag are not silently reinterpreted. Targets restricted to recent
returns (e.g. age < 0.1 s) do not reduce the encoder's ten-sweep input.

The configured BEV bounds are x/z `[-50,50]`, y `[-5,5]`, with centroid offset
`(0,1,0)`, in the reference-camera frame. `Vox_util` applies that centroid; do
not add it a second time. Grid `(Z,Y,X)=(200,8,200)`, horizontal cell size 0.5 m,
vertical cell size 1.25 m. Volumes are `[B,C,Z,Y,X]`, BEV maps `[B,C,Z,X]`.
Use `Vox_util.Ref2Mem/Mem2Ref` for exact center/index conventions.

## Features, targets and outputs

```python
import torch
from dd2414.models.multisweep import prepare_bevcar_inputs
from dd2414.interfaces import extract_features

# model is a caller-created, strictly loaded research model on device.
model.eval()
inputs = prepare_bevcar_inputs(legacy_batch, device)
with torch.no_grad():
    result = extract_features(model, inputs, location="fused_bev")
features = result.bev_features  # [B,128,200,200], NOT segmentation logits
```

The prepared tuple remains `(centered_rgb, pixel_from_cameras,
camera0_from_cameras, vox_util, (voxel_features, coordinates, counts))`.
Voxel tensors must come from the existing adapter, not an inferred replacement.
`camera_bev` reads the existing fixed height-mean before learned radar fusion;
`fused_bev` reads the shared fused feature. Both have 128 channels. Image features
retain the model's packed camera shape `[B*S,128,h,w]`. `depth_prediction` may be
None: metric-profile geometry is retained only when explicitly requested by
the training objective, not invented by this API. The adapter preserves the
caller's train/eval, autograd and autocast policies. A probe returns one-channel
vehicle logits; motion readouts return `(vx,vz)` in camera-0, m/s.

DINO features and radar-supported soft splats are frozen pseudo targets, not
GT vehicle masks. Metric Depth Anything provides pseudo camera-z in metres;
the random metric profile uses 80 probability bins over 1–80 m. Its depth loss
is applied after calibrated sampling into the camera BEV volume and before
radar fusion. Both treatments retain the same probability lifting; only the
depth treatment adds pseudo-target supervision. Legacy relative/log-depth
profiles are different implementations and are not silently converted.

## Metrics and state

`dd2414.evaluation.segmentation.official_counts` uses the original
`logits.sigmoid().round()`, GT/valid `>0.5`, global TP/FP/FN/TN sums. Vehicle IoU
is TP/(TP+FP+FN), not a per-frame average or multiclass mIoU. Existing historical
probe metrics with `>=0.5` remain distinct at exactly zero logits. Motion EPE
and intervention loss differences are auxiliary; neither is segmentation IoU.

Only tensor/state checkpoints are supported. Strict loading and recorded
source/config hashes are preserved. See [compatibility](integration/COMPATIBILITY.md)
for why exact historical optimizer/RNG/sampler resume uses its original source.
