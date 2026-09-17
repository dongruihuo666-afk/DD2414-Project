# Radar encoder data audit

## Scope

This audit is deliberately before model implementation. It inspects the exact
19-channel radar tensor that the current Simple-BEV loader returns, using the
first 10 fixed nuScenes mini training samples, one sweep, and radar filters
disabled to match the existing smoke/DINO experiments.

Run:

```bash
./scripts/run_radar_field_audit.sh
```

The sample contains 4,200 valid returns in total, 398-432 per frame (mean 420).
Every retained scalar is finite.

## The 19 fields

| Index | Field | Meaning | Initial encoder decision |
| ---: | --- | --- | --- |
| 0-2 | `x,y,z` | Point position after loader transformation | Use after ROI filtering and normalization |
| 3 | `dyn_prop` | Discrete dynamic-state code | Optional mask/embedding, not a real number |
| 4 | `id` | Radar cluster identifier | Do not use as a physical feature |
| 5 | `rcs` | Radar cross section / return strength | Use with clipping and robust scaling |
| 6-7 | `vx,vy` | Raw planar velocity | Skip initially |
| 8-9 | `vx_comp,vy_comp` | Ego-motion-compensated planar velocity | Use only after frame rotation is fixed |
| 10 | `is_quality_valid` | Discrete quality flag | Use as a mask; constant 1 in this subset |
| 11 | `ambig_state` | Doppler ambiguity code | Use as a mask/weight |
| 12-13 | `x_rms,y_rms` | Quantized uncertainty codes | Optional later embedding/weight |
| 14 | `invalid_state` | Cluster validity code | Use as a mask/weight |
| 15 | `pdh0` | Quantized false-alarm probability code | Use as a quality weight |
| 16-17 | `vx_rms,vy_rms` | Quantized velocity uncertainty codes | Optional later embedding/weight |
| 18 | `time_lag` | Difference from reference radar timestamp | Use after normalization |

These are not 19 interchangeable continuous measurements. IDs and state/RMS
codes should not be blindly standardized and passed to an MLP.

## Measured ranges and consequences

- Planar range has a 1st/99th percentile of 6.48/168.34 m. Many returns are
  outside the current 100x100 m BEV. Filter to the BEV ROI before a point MLP;
  otherwise extreme unused points still affect activations and compute.
- RCS has a 1st/99th percentile of -5.0/27.5 and a maximum of 42. Clip to robust
  percentiles before standardization.
- Compensated speed magnitude is below 7.61 m/s for 99% of returns, but isolated
  components reach 54.9 and -33.1 m/s. Velocity also needs robust clipping.
- Time lag lies between -0.0543 and 0.0584 s even with one sweep because the five
  radar timestamps are asynchronous relative to `RADAR_FRONT`. A signed time
  feature is therefore meaningful; it is not always nonnegative.
- `is_quality_valid` is 1 for 100% of these returns and cannot filter them by
  itself. Only 54.6% have `invalid_state == 0`, and 88.2% have unambiguous
  Doppler (`ambig_state == 3`). The current loader has disabled default filters.
- 46.7% have compensated speed magnitude below 0.1 m/s. Dynamic-state codes
  `[0,2,6]` mark 16.5% as moving/oncoming/crossing-moving, but this code must not
  be used as independent evaluation ground truth if it becomes a training cue.

## Critical frame issue before using velocity

`nuscenesdataset.get_radar_data()` applies `current_pc.transform(trans_matrix)`
to merge five radar sensors. In the installed nuScenes devkit,
`PointCloud.transform()` transforms only `points[:3]`, i.e. XYZ. It does not
rotate rows 6-9 containing raw or compensated velocity.

Consequently the current merged tensor has common-frame positions but velocity
components that were not rotated by the same sensor-to-reference transform.
Velocity magnitude remains useful because rotation preserves magnitude, but
directional `vx_comp/vy_comp` is not consistent across the five sensors.

Before feeding velocity direction to a radar encoder:

1. enable the tested sensor-to-reference-ego velocity rotation before the five
   radar tensors are concatenated; and
2. if the encoder works in Simple-BEV's reference-camera frame, rotate the
   vectors once more into that same frame as XYZ.

### Implemented optional correction and test

The correction is now available as `rotate_radar_velocity=True` in
`get_radar_data()`, `NuscData`, and `compile_data()`. Its default is `False`, so
official legacy checkpoints and previous baseline commands retain the original
behavior. The standalone point encoder test opts in explicitly.

Run:

```bash
./scripts/run_radar_velocity_test.sh
```

The standalone test verifies:

- a synthetic 90-degree rotation maps `(vx,vy)` to `(-vy,vx)` exactly;
- translation is never applied to a velocity vector;
- real-sample XYZ and every nonvelocity metadata row remain unchanged;
- raw and compensated velocity magnitudes are preserved (maximum measured
  compensated-speed error `3.4e-7` m/s); and
- all 200 returns above 0.1 m/s change direction when the precise calibration
  rotations are applied.

The median real-sample direction change is 174.9 degrees because 118/200 of
those nonstatic returns come from the two rear radars, whose calibrated yaws are
approximately +174.7/-175.0 degrees. Front radar yaw is approximately -0.3
degrees and the side radar yaws are approximately +/-90 degrees. This confirms
that the large median is caused by sensor mounting geometry.

![Radar velocity frame correction](artifacts/radar_velocity_rotation.png)

## Proposed first numeric input

After ROI filtering, use a small continuous vector:

```text
[normalized reference-frame x, y, z,
 robust-scaled RCS,
 rotated/clipped vx_comp, vy_comp,
 normalized signed time_lag]
```

Keep quality information separate as masks or weights. The first encoder can be:

```text
7 numeric values -> MLP(32) -> MLP(64)
                 -> voxel mean/max pooling
                 -> 64-channel radar BEV
```

## Implemented minimal encoder

`nets/radar_encoder.py` now implements this exact standalone path:

```text
five radar sensors
  -> rotate velocity into one reference-ego frame
  -> transform XYZ and velocity into reference-camera/BEV axes
  -> select and normalize [x,y,z,RCS,vx_comp,vz_comp,time_lag]
  -> point MLP 7->32->64
  -> 3D voxel mean and max pooling
  -> collapse height
  -> (B,64,200,200) radar BEV
```

No segmentation map, vehicle box, center, offset, or other human annotation is
read by the encoder. The three discrete fields `is_quality_valid`,
`ambig_state`, and `invalid_state` are used only as a conservative point mask,
not as continuous MLP inputs.

Run:

```bash
./scripts/run_radar_point_encoder_test.sh
```

On the first real mini frame, 403 padded-tensor input points became 125
in-range, high-confidence points, occupying 123 3D voxels and 123 BEV cells.
The result shape was `(1,64,200,200)`. The test also passed finite-value, empty
input, camera-frame position/velocity transform, point-order invariance,
same-voxel collision pooling, and nonzero finite-gradient checks.

![Minimal radar point encoder](artifacts/radar_point_encoder_test.png)

This proves the radar branch can produce a learnable BEV tensor with correct
geometry and gradients. It does not yet prove useful learned semantics or
motion because its weights are random and it has not been trained by the DINO
loss.

## Camera-BEV integration

The encoder is now connected to the main `Segnet` path behind the independent
`use_radar_encoder` option. The data flow is:

```text
six images -> image encoder -> camera BEV volume -> flatten height --+
                                                                  concat
radar points -> 7-field point encoder -> 64-channel radar BEV -----+
                                      -> 3x3 BEV fusion -> decoder
```

This path is intentionally separate from the official `use_radar` and
`use_metaradar` flags, so the official checkpoints keep their original tensor
shapes and behavior. Run the integration test with:

```bash
./scripts/run_radar_fusion_test.sh
```

On one real nuScenes mini frame, the test measured:

- radar encoder output `(1,64,200,200)`;
- fused camera-radar feature `(1,128,200,200)`;
- finite radar-encoder gradient norm `0.031452`;
- valid forward output when all radar points were replaced by padding;
- successful checkpoint save and reload; and
- peak allocated CUDA memory `3.440 GiB`, below the available 8 GiB budget.

![Learned radar encoder fused with camera BEV](artifacts/radar_encoder_fusion_smoke.png)

This is an engineering integration check, not an accuracy claim. The network
is randomly initialized in this test, so its prediction and IoU are not
meaningful yet. The next experiment should decide how the fused feature is
trained without box-derived labels.

## BEVCar-shaped voxel input: one-frame geometry audit

To evaluate a possible [BEVCar](https://github.com/robot-learning-freiburg/BEVCar)
radar-encoder transplant without changing the working lightweight encoder,
`nets/bevcar_voxel_adapter.py` now prepares the three tensors expected by the
released [VoxelNet forward
interface](https://github.com/robot-learning-freiburg/BEVCar/blob/main/nets/voxelnet.py):

```text
reference-camera radar points (B,R,19)
  -> Simple-BEV Ref2Mem coordinates on the same (Z,Y,X)=(200,8,200) grid
  -> quality/ROI filter -> group by 3D voxel
  -> point features (B,K,16,7), coordinates (B,K,3) in (z,y,x) order,
     occupied-voxel count (B,)
```

The seven prototype channels are `[x,y,z,raw_vx,raw_vz,RCS,time_lag]` in the
reference-camera frame. The [BEVCar paper](https://arxiv.org/html/2403.11761)
explicitly describes position, *uncompensated* planar velocity and RCS, while
the released VoxelNet declares seven input channels. We use time lag as an
explicit seventh channel for this geometry-stage prototype; **we have not
confirmed that its feature semantics match BEVCar's released preprocessor or
pretrained checkpoint**. The current quality filter matches our lightweight
encoder, not necessarily BEVCar's published training setup. If a voxel has
more than 16 points, this audit keeps the first 16 and reports the overflow;
the paper describes random sampling instead. Exceeding `max_voxels` raises an
error rather than silently discarding spatial evidence.

Run the CPU-only check:

```bash
bash scripts/run_bevcar_voxel_adapter_test.sh
```

It reads the first fixed mini training key frame directly from nuScenes files,
so it also runs in an environment where the existing `VizData` constructor's
unconditional `.cuda()` is unavailable. The frame token is
`cd9964f8c3d34383b16e9c2997de1ed0`, matching the 403-return frame used
in the lightweight-encoder audit. Results: 403 input returns, 251 inside the
BEV ROI, 125 after the same quality filter, 123 occupied 3D voxels, zero
truncated points and zero BEV-cell mismatches against the point-coordinate
map. The synthetic checks cover collision grouping, empty/batched samples,
voxel limits and `(z,y,x)` axis direction.

![BEVCar-shaped voxel input geometry audit](artifacts/bevcar_voxel_adapter_audit.png)

This is **input preparation and coordinate validation only**: no BEVCar
weights, network forward/backward, GPU-memory measurement, or accuracy result.
Before attempting any pretrained BEVCar checkpoint, its exact preprocessing
and field order must be established. The next bounded step is an isolated
random-weight VoxelNet smoke test with output shape, gradient and memory
checks; preserve the lightweight path as a separate comparison baseline.
