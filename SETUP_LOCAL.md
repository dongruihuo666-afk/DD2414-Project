# DD2414 Simple-BEV local setup

## Current Ubuntu trainval host (2026-10-05)

The current `/home/students2026/dd2414` checkout uses the `bev` environment in
`${HOME}/miniconda3`: Python 3.10.14, torch 2.5.1, torchvision 0.20.1, CUDA 12.4,
and one RTX 3090 Ti (24 GB). Full labeled `v1.0-trainval` is installed at
`${HOME}/datasets/nuscenes` (700 train / 150 val scenes, 28,130 / 6,019 samples).
Mini was removed on this host; the historical mini commands below need a
separate mini installation. DINOv2 source/weights are cached. Official BEVCar
source is installed at `external/BEVCar`, detached at commit
`29cacda3bc5416d47428c1d0f017527acad34f90`; project experiments import only
its `nets/voxelnet.py` and do not load a BEVCar checkpoint or full model.

Use `configs/trainval.env` and `bash scripts/run_trainval.sh dual-teacher` to
preview the full-data configuration without starting anything. The launcher
activates Conda itself when explicitly executed. See [TRAINVAL_RUN_PLAN.md](TRAINVAL_RUN_PLAN.md)
for the ordered smoke tests, bounded experiments, full supervised baseline and
remaining streaming self-supervised work.

## Historical mini host and repository

- Host: Windows 11 + WSL2, Ubuntu 22.04.5 LTS, kernel `6.18.33.2-microsoft-standard-WSL2`.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB VRAM, Windows driver 591.91.
- Conda environment: `simplebev`, Python 3.10.21. The helper scripts assume
  `${HOME}/miniconda3` by default; set `CONDA_ROOT` if Conda is installed elsewhere.
- Upstream repository: `https://github.com/aharley/simple_bev`.
- Upstream commit: `be46f0ef71960c233341852f3d9bc3677558ab6d`.
- Development branch: `dd2414-mini-baseline`.

The Codex filesystem sandbox blocks NVML, so `nvidia-smi` and CUDA tests must be
run in a WSL execution context with GPU access. The actual WSL environment was
verified to report `torch.cuda.is_available() == True`.

## Installed versions

The existing PyTorch installation was preserved. The Lyft SDK is intentionally
not installed because this phase only uses nuScenes.

| Package | Version |
| --- | --- |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| numpy | 1.26.4 |
| nuscenes-devkit | 1.2.0 |
| efficientnet-pytorch | 0.7.1 |
| fire | 0.4.0 |
| pyquaternion | 0.9.9 |
| opencv-python-headless | 4.11.0.86 |
| matplotlib | 3.10.9 |
| scikit-learn | 1.1.2 |
| scikit-image | 0.19.3 |
| scipy | 1.15.3 |
| Shapely | 2.0.7 |
| tensorboard / tensorboardX | 2.10.0 / 2.2 |
| protobuf | 3.19.4 |

NumPy is pinned to 1.26.4 because this 2022 codebase and its older scientific
dependencies use APIs that changed in NumPy 2.x. `nuscenes-devkit` was set to
1.2.0 so it works with a modern wheel-based Matplotlib installation on Python
3.10. `pip check` reports no broken requirements.

## Historical mini dataset

For the historical mini experiments, the official mini archive was downloaded:

- URL: `https://www.nuscenes.org/data/v1.0-mini.tgz`
- Size: 4,167,696,325 bytes (3.88 GiB)
- Local SHA-256: `4478a1f017b3cfd64d6f114ba44ff9fc482174d3d0bb90dcd3e1646cf9aff50f`
- Local test root: `${HOME}/datasets/nuscenes` (override with `NUSCENES_ROOT`).

The checksum above records the verified local file; it is not presented as a
publisher-supplied checksum. `gzip -t` passed before extraction. The directory
contains:

```text
${NUSCENES_ROOT}/
├── maps/          # 4 map files
├── samples/       # 4,848 keyframe sensor files
├── sweeps/        # 26,358 non-keyframe sensor files
└── v1.0-mini/     # 13 JSON tables
```

The devkit loads 10 scenes, 404 samples, and 31,206 sample-data records.

## Historical mini reproduction commands

Run from the repository root. Paths can be changed with `CONDA_ROOT`,
`CONDA_ENV`, `NUSCENES_ROOT`, `OUTPUT_DIR`, `LOG_DIR`, and `CKPT_DIR`.

```bash
./scripts/check_environment.sh
./scripts/smoke_test_mini.sh --data-only
./scripts/smoke_test_mini.sh
./scripts/smoke_test_mini.sh --use-radar --use-metaradar
./scripts/train_mini_short.sh
./scripts/run_meeting_demo.sh
./scripts/run_mini_subset_eval.sh
./scripts/run_baseline_overfit.sh
./scripts/run_dinov2_bev_demo.sh
./scripts/run_semantic_distill_smoke.sh
./scripts/run_dinov2_mini4.sh
./scripts/run_radar_field_audit.sh
./scripts/run_radar_velocity_test.sh
```

Official camera-only checkpoint inference at a low smoke-test resolution:

```bash
./scripts/smoke_test_mini.sh \
  --encoder-type res101 \
  --checkpoint 'checkpoints/8x5_5e-4_rgb12_22:43:46' \
  --no-backward \
  --visualization-name mini_camera_pretrained.png
```

The default smoke configuration is batch size 1, workers 0, six cameras,
one radar sweep, EfficientNet-B0, AMP, and an image size of 112x192. The
original 200x200 BEV grid is retained. Reduced image dimensions are aligned to
16 pixels because the encoder skip connection requires exact stride-8 and
stride-16 alignment.

## Verified results

- One mini batch: six images, intrinsics/extrinsics, ego pose, BEV labels,
  lidar, and radar all loaded. The first batch had 403 valid radar points.
- Camera-only random-initialized smoke: forward 0.987 s; backward + optimizer
  step 0.457 s; finite loss 12.439; peak CUDA allocation 3.417 GiB.
- Camera + 16-channel metaradar smoke: forward 1.100 s; backward + step
  0.466 s; finite loss 14.047; peak CUDA allocation 3.417 GiB.
- Official camera checkpoint: safe load with `weights_only=True`; no-grad
  forward 0.604 s; peak CUDA allocation 3.501 GiB; one-batch IoU 0.0852.
  This is a low-resolution functionality check on mini, not a fair reproduction
  of the trainval checkpoint's published result.
- Official camera+radar checkpoint: safe load with `weights_only=True` and a
  1-sweep metaradar no-grad forward; 0.657 s, 3.501 GiB peak allocation, and
  one-batch IoU 0.2757. The checkpoint was trained with five sweeps, so this is
  interface validation rather than its reported configuration.
- Final five-step camera training: finite losses 9.538, 9.574, 9.252, 9.471,
  and 9.033; peak CUDA allocation 3.577 GiB. The step-5 checkpoint was safely
  loaded into a fresh model path and completed another forward.
- Visual outputs are in `artifacts/`; generated checkpoints are in
  `artifacts/smoke_checkpoint/` and `checkpoints/`.
- Frozen DINOv2 ViT-S/14 extraction and radar-anchored soft-target generation:
  `(6,384,14,24)` camera features, 403 camera-visible radar anchors, 16,064
  supervised BEV cells, 1.534 GiB peak allocation. The xFormers-unavailable
  warnings are expected; PyTorch's fallback attention completed successfully.
- Self-supervised semantic one-sample smoke: 20-step confidence-masked cosine
  loss decreased from 0.985776 to 0.203925 (final eval), compressor gradients
  were finite, checkpoint reload passed, and peak allocation was 3.540 GiB.
- Four-sample DINOv2 distillation with a randomly initialized Simple-BEV
  student and no supervised BEV checkpoint reduced mean pre/post DINO loss from
  1.023 to 0.196. The 60-step first/last-cycle means were 0.920/0.206; target
  generation took 2.9 s, training and evaluation 10.3 s, and peak allocation was
  4.18 GiB. No segmentation, center, offset, or box loss was used.
- Radar field audit on 10 fixed single-sweep mini samples: 4,200 finite valid
  returns, 398-432 per sample. It records robust ranges and categorical states,
  and identifies that the loader transforms XYZ but does not rotate velocity
  rows when merging the five radar sensors.
- Optional radar velocity rotation passed a synthetic 90-degree vector test and
  a 403-point real-sample test. XYZ and nonvelocity metadata are unchanged,
  compensated speed magnitude differs by at most `3.4e-7` m/s, and direction
  changes match the five calibrated sensor yaws. The option defaults to off to
  preserve legacy checkpoint behavior.
- The standalone minimal radar point encoder passed real mini and synthetic
  acceptance tests. It maps seven continuous fields through a point MLP and
  mean/max voxel pooling to `(1,64,200,200)`. On the first frame it retained
  125/403 points in 123 voxels; empty input, point-order invariance, collision
  pooling, camera-frame transforms, finite output, and backward gradients pass.
- A post-change official camera+radar forward regression passed with IoU 0.2806
  on the same low-resolution mini sample.
- A fixed first-10 validation subset with 112x192 images and five radar sweeps
  produced mean IoU 0.121 +/- 0.039 for the official camera checkpoint and
  0.291 +/- 0.043 for the official camera+radar checkpoint. Radar was higher
  on 10/10 samples. After one untimed warm-up per model, mean forward time was
  0.089 s vs. 0.092 s and both peaked at 3.52 GiB. This is a tiny functionality
  study, not the paper benchmark.
- A four-sample, 50-update supervised camera+radar overfit reduced the mean
  positive task loss over a four-step cycle from 12.698 to 0.790 (93.8%) and
  raised mean IoU on the same memorized samples from 0.273 to 0.803. Training
  plus before/after evaluation took 8.2 s and peaked at 4.00 GiB. This is a
  training-loop sanity check, not validation performance.

## Compatibility changes and reasons

1. `nuscenesdataset.py`: Lyft SDK import is optional. Previously every nuScenes
   import failed unless the unrelated Lyft package was installed. Lyft requests
   now fail with a targeted message; nuScenes behavior is unchanged.
2. `nuscenesdataset.compile_data()`: accepts the standard official layout
   `<root>/v1.0-mini` while retaining the repository's legacy
   `<root>/mini/v1.0-mini` fallback.
3. `nets/segnet.py`: normalization constants are non-persistent buffers rather
   than tensors allocated with `.cuda()` during construction. Numerical model
   behavior is unchanged and device movement is now correct.
4. `nets/segnet.py`: torchvision's current `weights=` API is used. A
   `pretrained_backbone` switch avoids downloading ImageNet weights immediately
   before loading a complete checkpoint. The default remains pretrained.
5. `train_nuscenes.py`: `run_model()` accepts either a direct model or
   `DataParallel`, can optionally return predictions for smoke visualization,
   aligns low-resolution inputs to 16 pixels, and reports peak GPU memory.
6. `saverloader.py`: `torch.load(..., weights_only=True)` is explicit for safe
   PyTorch 2.7 checkpoint loading. Both official archives contain ordinary
   model/optimizer state dictionaries.
7. `vis_nuscenes.py`: map lookup uses the loader's actual resolved data root.
8. `nets/segnet.py`: an optional `return_shared_bev=True` path exposes the
   `(B,128,200,200)` fused feature before the supervised decoder plus the
   pre-projection `(B*S,128,Hf,Wf)` image features for image-level distillation.
   The default five-output forward path is unchanged and passed a checkpoint
   regression.
9. `scripts/dinov2_bev_demo.py`: creates and caches label-free frozen-teacher
   features plus radar-anchored soft-region semantic targets.
10. `scripts/semantic_distill_smoke.py`: adds a training-only semantic head and
    tests masked cosine distillation, gradient flow, saving, and reload on one
    deterministic sample.
11. `scripts/eval_mini_subset.py`: evaluates both official checkpoints on the
    same fixed mini validation subset, saves CSV/NPZ outputs, and selects the
    highest, middle, and lowest radar-IoU-delta examples for an honest summary
    board.
12. `scripts/baseline_overfit_mini.py`: cycles over four fixed mini training
    samples for 50 updates, records positive task losses and gradients, saves a
    checkpoint, and visualizes prediction changes. It uses a conservative AMP
    initial scale and holds the checkpoint's learned uncertainty weights fixed
    so the plot reflects segmentation, center, and offset task losses.
13. `scripts/dinov2_mini4_experiment.py`: caches four radar-anchored DINOv2
    targets and trains a randomly initialized student for 60 alternating
    updates without loading a supervised BEV checkpoint or using box-derived
    losses. Float32 is used for the tiny pre/post evaluation because an entirely
    random ResNet has uncalibrated BatchNorm statistics that can overflow fp16
    in eval mode; training remains AMP.
14. `scripts/radar_field_audit.py`: exports statistics for all 19 radar fields,
    quality-state proportions, robust histograms, and first-encoder feature
    recommendations without changing or training the model.
15. `nuscenesdataset.py`: an opt-in `rotate_radar_velocity` data flag rotates
    raw and compensated planar vectors with the sensor-to-reference rotation,
    excluding translation. The default is `False` for backward compatibility.
16. `scripts/test_radar_velocity_rotation.py`: tests exact synthetic rotation,
    real-data invariants, finite values, magnitude preservation, calibration
    direction changes, and creates a presentation visualization.
17. `nets/radar_encoder.py`: adds an independent seven-field point MLP with
    quality/ROI masking, mean/max 3D voxel pooling, and height-collapsed radar
    BEV output; it is deliberately not wired into the official model yet.
18. `scripts/test_radar_point_encoder.py`: validates the encoder on synthetic
    cases and a real mini frame, including geometry, permutation, empty-input,
    collision, and gradient checks, then exports a presentation image and NPZ.

The first attempted 112x200 EfficientNet forward failed while concatenating a
25-wide skip map with a 24-wide upsampled map. Aligning the width to 192 fixes
the implicit encoder stride constraint without deleting or bypassing features.
