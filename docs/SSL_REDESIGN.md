# Camera + radar self-supervised BEV pretraining (redesign)

Branch `ssl-radar-camera`, based on upstream Simple-BEV (`main`). Uses no labels and no LiDAR during pretraining. Teachers are frozen DINOv2 (semantics) and Depth-Anything-V2 metric (geometry), plus held-out radar.

## What was wrong in the previous experiment branches

| Problem | Effect | Fix here |
|---|---|---|
| The BEV DINO target is the feature of the pixel a radar point projects to. Simple-BEV lifting already copies that pixel's feature along the whole ray. | The camera alone satisfies the loss; radar only selects the loss mask. Recorded: loss with no radar ≤ loss with radar. | Semantic target placed at a **metric 3D location** using radar-scale-aligned teacher depth (`ssl_pretrain/targets.py: semantic_bev_target`). |
| The radar motion target is the radar velocity at radar cells, and the student gets the same radar as input. | Identity task. | Velocity is predicted at **held-out** returns only, as a radial (Doppler) loss (`split_radar`, `radial_velocity`). |
| Radar quality fields are ignored (`x/y_rms`, `vx/vy_rms`, `pdh0`, `ambig_state`, `invalid_state`); raw velocity is used. | No notion of which returns to trust. | `nets/radar_uncertainty.py`: decoded ARS408 stds give an anisotropic Gaussian splat per return; `pdh0`/ambiguity/validity give a confidence prior; enum embeddings are also used. Older sweeps are motion-compensated with Doppler velocity, and their lateral std grows with age. |
| Fusion is concat + conv. | Radar is easy to ignore. | `nets/fusion.py`: gates driven by camera and radar confidence maps, plus **modality dropout** in pretraining. |
| No loss on a single modality. | No cross-modal transfer. | Separate camera / radar / fused heads. Camera→radar: the radar branch must predict DINO semantics. Radar→camera: the camera branch must predict held-out radar occupancy. |
| The depth head is used as a feature gate (features ×≈1/80). Depth CE is in BEV voxels. The VKITTI metric scale is never aligned. | Unstable geometry. Depth CE rose during training. | LSS-style density lifting. Depth CE is computed **in image space**. Teacher depth scale is aligned per camera to radar returns. |
| The probe is 1×1 on height-averaged camera BEV. | Official Simple-BEV scores 2.6% IoU on it: the probe measures nothing. | `probe_bev.py`: the same probe on the 128-d fused/compressor BEV for every model (ours, official Simple-BEV rgb/rad, ImageNet), plus a frozen-backbone decoder probe and a low-label fine-tune. |
| Only `encoder.*` loaded from the official checkpoint; no augmentation; batch 1; 5 epochs. | The downstream number is set by the recipe, not the pretraining. | The official Simple-BEV recipe (`train_nuscenes.py`) is used with `--model_type=ssl --init_ckpt --label_frac`. |

## Files

```
nets/radar_uncertainty.py   ARS408 decode tables, RadarUncertaintyEncoder, radar ego->cam0
nets/fusion.py              UncertaintyGatedFusion
nets/ssl_bev.py             SSLBEVNet (camera depth lifting, radar, fusion, SSL heads, decoder)
ssl_pretrain/teachers.py    DINOv2 (registers) + Depth-Anything-V2 metric
ssl_pretrain/targets.py     radar split, depth alignment, semantic/occupancy/velocity targets
ssl_pretrain/losses.py      losses + default weights
ssl_pretrain/step.py        one pretraining step (geometry -> teachers -> targets -> losses)
train_ssl.py                pretraining
probe_bev.py                linear / decoder / model evaluation, radar controls
ssl_runner.sh               full experiment matrix
tests/                      CPU tests (synthetic 6-camera rig)
```

Small upstream fixes:
- `utils/vox.py`: cast bounds to float, needed for newer torch.
- `nets/segnet.py`: `return_bev` hook; no hard `.cuda()` in `__init__`.
- `train_nuscenes.py`: SSL model type, `--label_frac`, `--freeze_backbone`, a CPU-safe loss accumulator.
- `nuscenesdataset.py`: optional radar velocity rotation into the ego frame. Ported from the student branch. It is off by default because official checkpoints expect unrotated velocities.

## Setup

1. The environment from the main README, plus the Depth-Anything-V2 repository and a **metric outdoor (VKITTI)** checkpoint, e.g. `depth_anything_v2_metric_vkitti_vits.pth`.
2. DINOv2 loads through `torch.hub` (`dinov2_vits14_reg`). Pass `--dino_hub_dir` for an offline copy.
3. `python -m pytest -q tests` should pass (CPU, about 15 s).

## Run order

1. **Sanity check before any long run.** Pretrain on `--dset=mini` for about 2k iterations. In TensorBoard, check that:
   - `depth_scale_mean` sits roughly in 0.8–1.3 and `depth_scale_anchored_frac` is high;
   - `sem_coverage` / `occ_known_frac` are nonzero;
   - every loss term decreases;
   - `gate_rad` does not collapse to 0.
2. `./ssl_runner.sh baselines`: the official Simple-BEV and ImageNet numbers under the new probe.
3. `./ssl_runner.sh pretrain`, then `./ssl_runner.sh probe <ckpt> <tag>` for each run.
4. `./ssl_runner.sh lowlabel <ckpt>`.

Radar reliance: every fused probe is also evaluated with `--radar_mode=empty` and `--radar_mode=wrong`. A model that really uses radar must lose IoU in both.

## Expectations and caveats

- Official Simple-BEV features are **supervised on vehicle segmentation**. Beating them with a *linear probe* from label-free pretraining is unlikely. The fair targets are:
  - beating ImageNet init and the old DINO setup at every readout;
  - beating Simple-BEV in the 1% and 10% label regimes;
  - showing a clear empty/wrong-radar drop.
- ARS408 table values (`RMS_TABLE`, `PDH0_P_REAL`) follow the Continental cluster-quality spec. They act only as a soft prior; learned embeddings of the raw codes are also used. Verify them against the spec sheet if exact values matter.
- `x_rms` / `y_rms` are sensor longitudinal/lateral. They are applied as radial/tangential with respect to cam0, which is approximate for side and rear radars.
- Teacher depth beyond about 60 m, and sky, are masked. Radar has no elevation, so scale alignment uses a robust median over all anchors per camera.
- Low-label runs with `--max_iters=25000` at 1% labels will overfit. Reduce the iterations or select on a held-out split.
