# Tensor and Python interface

`dd2414.data.batch.as_batch(sequence)` is a zero-copy named view of the
18-field `NuScenesData` collated tuple. Construction does not validate, move,
normalize or reorder values. `.validate()` is an explicit boundary shape check.

| Position/name | Shape and interpretation |
| --- | --- |
| 0 images | float32 `[B,T,S,3,H,W]`, RGB `[0,1]` |
| 1 camera_rotations | float32 `[B,T,S,3,3]`, camera-to-ego rotation |
| 2 camera_translations | float32 `[B,T,S,3]`, camera origin in ego, metres |
| 3 intrinsics | float32 `[B,T,S,4,4]`, homogeneous resized/cropped pixel calibration |
| 4–7 lidar_current, lidar_current_extra, lidar, lidar_extra | Legacy padded channel-first arrays |
| 8–11 boxes, visibility, track_ids, scores | Legacy padded labels/identities; not predictions |
| 12 vehicle_mask, 13 valid_mask, 14 centers | float32 `[B,T,1,Z,X]`, supervised targets/validity |
| 15 offsets | float32 `[B,T,2,Z,X]`, grid-cell offset targets |
| 16 radar | float32 `[B,T,19,R]`, `R=700*nsweeps` zero padded |
| 17 ego_pose | Original homogeneous ego pose |

Research paths normally use time zero (`T=1`). CPU loader outputs are explicitly
moved/cast by preparation functions, not the view. Student RGB is centered by
subtracting 0.5; DINO RGB is separately resized and normalized by ImageNet mean
`(.485,.456,.406)` / std `(.229,.224,.225)`. Do not apply either twice.

Deterministic camera order before the reference swap is FRONT_LEFT, FRONT,
FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT. `refcam_id=1`, shuffling disabled,
swaps slots zero/one: FRONT then FRONT_LEFT, not a cyclic rotation. Official
training may shuffle/reference-select; always retain paired calibration.
`A_from_B` maps B column coordinates to A. Rotations/translations give
ego_from_camera; inverse gives camera_from_ego. Camera-0 is x-right, y-down,
z-forward. Pixels and metres are distinct.

Radar fields are `x,y,z,dyn_prop,id,rcs,vx,vy,vx_comp,vy_comp,is_quality_valid,
ambig_state,x_rms,y_rms,invalid_state,pdh0,vx_rms,vy_rms,age`. Coordinates are
metres; velocities metres/second; last field seconds from current reference
timestamp minus sensor timestamp. Five physical sensors each contribute up to
`nsweeps` temporal records. Ten sweeps does not mean ten physical radars.
Past XYZ uses sensor/past ego/current ego transforms. This compensates ego poses,
not moving-object GT trajectories. The existing opt-in velocity rotation rotates
both raw and compensated pairs, without translation. After preparation pairs
store camera-0 `(vx,vz)`; motion targets use compensated channels 8:10.
Do not reinterpret historical unrotated runs as rotated ones. XYZ-zero padding,
quality/finiteness/age checks remain objective-specific. A recent-return target
filter does not reduce encoder sweep input. Per-branch voxel limits are preserved.

Configured BEV bounds x/z `[-50,50]`, y `[-5,5]`, centroid `(0,1,0)`, grid
`(Z,Y,X)=(200,8,200)`: horizontal 0.5 m, vertical 1.25 m. Vox_util applies the
centroid and cell-center conventions; use Ref2Mem/Mem2Ref. Volumes are
`[B,C,Z,Y,X]`, BEV tensors `[B,C,Z,X]`. Do not add centroid twice.

```python
from dd2414.data.batch import as_batch
from dd2414.interfaces import prepare_inputs, extract_bev
view = as_batch(batch)
inputs = prepare_inputs(batch, device)
# Caller owns eval/train, no_grad and autocast policy.
features = extract_bev(model, inputs)
```

Prepared inputs retain `(centered_rgb, pixel_from_cameras,
camera0_from_cameras, vox_util, (voxel_features, coordinates, counts))`.
Fused features are `[B,128,Z,X]`, NOT vehicle logits or a depth GT map.
One-channel probe/segmentation logits and 2-channel camera-plane velocity
outputs retain their existing branch APIs.

This historical hybrid wrapper exposes fused features via `shared_bev(*inputs)`.
The integration API supports that existing location only; a camera-before-fusion
API is not invented. The historical mean-per-frame IoU and global IoU use different aggregation
and threshold conventions; see the metric note in [the experiment](EXPERIMENT.md).

GT vehicle labels, frozen DINO features, radar-derived soft targets and depth
pseudo-labels are different evidence. Low reconstruction loss is not vehicle
IoU; GT-box motion readouts are not object detection/NDS. Metric/relative depth
profiles remain distinct. Strict tensor/state checkpoint schemas, names and
shapes are preserved. See [recovery](REPRODUCIBILITY.md) for exact state resume.
