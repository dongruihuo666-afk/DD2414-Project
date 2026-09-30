# Simple-BEV model and data-flow notes

## What it solves

Simple-BEV predicts a bird's-eye-view representation of vehicles around the
ego car from six surround-view cameras, optionally augmented with lidar or
radar. In this repository the supervised outputs are a vehicle occupancy map,
instance-center heatmap, and per-cell instance-center offsets on a 200x200 BEV
grid covering roughly 100 m by 100 m.

## End-to-end tensor flow

1. `nuscenesdataset.compile_data()` creates `VizData` loaders. Each item from
   `VizData.get_single_item()` contains six resized RGB images, calibration,
   ego pose, lidar/radar sweeps, object boxes, and BEV targets.
2. `train_nuscenes.run_model()` removes the length-one time dimension and
   converts camera calibration into `pix_T_cams` and `cam0_T_camXs`.
3. `nets.segnet.Segnet.forward()` packs batch and camera axes. One of
   `Encoder_res101`, `Encoder_res50`, or `Encoder_eff` produces a 128-channel
   feature map for each camera. In the tested 112x192 setup its packed shape is
   `(6, 128, 14, 24)`.
4. `Vox_util.unproject_image_to_mem()` samples each 2D feature map into a common
   3D reference-camera voxel grid. The six camera volumes have shape
   `(B, 6, 128, 200, 8, 200)` and are combined with a masked mean over cameras.
5. Height is flattened into channels and `bev_compressor` produces
   `feat_bev` with shape `(B, 128, 200, 200)`.
6. `Decoder` is a ResNet-18-style BEV encoder/upsampler. It returns shared
   decoder features plus three heads: segmentation `(B,1,200,200)`, center
   `(B,1,200,200)`, and offset `(B,2,200,200)`.

The camera projection is geometry-driven rather than learned depth lifting:
known intrinsics and camera extrinsics determine which 2D feature is sampled
for every voxel. Empty projections are masked before averaging cameras.

## Radar read and fusion

`nuscenesdataset.get_radar_data()` reads all five nuScenes radar sensors using
`RadarPointCloud`, transforms current and previous sweeps into the current ego
frame, and appends sweep time lag. The result is 19 values per point: XYZ,
the 15 native non-position radar fields, and time lag.

`train_nuscenes.run_model()` converts radar XYZ to the reference-camera frame.
It builds either:

- a one-channel occupancy voxel grid (`use_radar=True`), or
- a 16-channel grid from the non-XYZ fields including Doppler-related velocity
  and time (`use_radar=True, use_metaradar=True`).

In `Segnet.forward()`, radar is collapsed/reshaped along the vertical axis,
concatenated with the camera volume after multi-camera unprojection, and fused
by `bev_compressor`. Camera-only and camera+radar therefore share the same BEV
decoder; only the compressor input channel count differs.

## Supervision and losses

This is supervised because the targets come from nuScenes human-annotated 3D
vehicle boxes:

- `NuscData.get_lrtlist()` selects vehicle annotations and transforms boxes.
- `NuscData.get_seg_bev()` rasterizes box footprints into vehicle occupancy and
  produces a visibility-based valid mask.
- `NuscData.get_center_and_offset_bev()` creates instance-center heatmaps and
  offsets from each occupied cell to its box center.

`train_nuscenes.run_model()` combines:

- masked, positive-weighted binary cross entropy for vehicle segmentation;
- balanced MSE for the center heatmap;
- masked L1 for the two-component instance offset.

Three learned scalar uncertainty weights rescale these losses. This is why the
baseline cannot be called self-supervised even though the image-to-BEV geometry
uses calibration rather than labels.

## Checkpoints and visualization

`saverloader.save()` writes model and optimizer state dictionaries;
`saverloader.load()` selects the largest numbered checkpoint and now uses
PyTorch's safe weights-only loader. `vis_nuscenes.py` is the original sequence
visualizer and saves camera, prediction, ground truth, radar, and lidar images.
For a one-batch reproducible collage, use `scripts/smoke_test_mini.py` through
`scripts/smoke_test_mini.sh`.

## Most useful extension seams

- The best task-neutral shared representation is `feat_bev` immediately after
  `bev_compressor` and before `Decoder` in `Segnet.forward()`.
- Per-camera teacher features can be produced alongside `feat_camXs_`, then
  projected with the same calibration and `Vox_util` geometry.
- Semantic distillation and radar motion heads should be auxiliary heads on
  `feat_bev`, leaving the current supervised decoder available for downstream
  fine-tuning and evaluation.
- Soft-region correspondence belongs after teacher/student features have been
  expressed in the same BEV grid and before loss reduction.

See `SELF_SUPERVISED_EXTENSION_PLAN.md` for the proposed research design.
For a short presentation and a one-command visual comparison, see
`PROJECT_PROGRESS.md` and `scripts/run_meeting_demo.sh`.
