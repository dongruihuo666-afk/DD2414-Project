# Multi-sweep double-distillation study

All scientific runs use 28,130 train frames and the same 6,019 validation frames.
Every comparison is single-seed and short-budget, not a convergence claim.
Official original-resolution references (448x800) are in the previous study; this queue also evaluates those checkpoints at 112x192.
New/legacy encoders use 112x192. A low-resolution evaluation of high-resolution-trained official weights is still not a matched training protocol.
The depth treatment changes BOTH supervision and lifting; isolate those components before attributing a gain to depth loss alone.

| Configuration | Status | Vehicle IoU (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| finish_existing_historical_single_sweep_finetuning | complete | 27.588 | 54.964 | 35.646 | 43.246 |
| recover_legacy_five_sweep_frozen_probe | complete | 5.117 | 5.214 | 73.320 | 9.736 |
| official_camera_112x192 | complete | 12.445 | 55.085 | 13.850 | 22.135 |
| official_camera_radar_5sweep_112x192 | complete | 22.862 | 71.225 | 25.188 | 37.215 |
| double_dino_5sweep_depth_none_pretrain_1epochs | complete | pending | pending | pending | pending |
| double_dino_5sweep_depth_none_frozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_none_finetune_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_none_unfrozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_none_object-motion_1epochs | queued | pending | pending | pending | pending |
| double_dino_10sweep_depth_none_pretrain_1epochs | queued | pending | pending | pending | pending |
| double_dino_10sweep_depth_none_frozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_10sweep_depth_none_finetune_1epochs | queued | pending | pending | pending | pending |
| double_dino_10sweep_depth_none_unfrozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_10sweep_depth_none_object-motion_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar_pretrain_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar_frozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar_finetune_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar_unfrozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar_object-motion_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar-dav2_pretrain_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar-dav2_frozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar-dav2_finetune_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar-dav2_unfrozen-probe_1epochs | queued | pending | pending | pending | pending |
| double_dino_5sweep_depth_radar-dav2_object-motion_1epochs | queued | pending | pending | pending | pending |

## Monitoring and reliability

The queue has no elapsed-time, log-idle or 24-hour kill rule. Long validation only emits a warning.
Failed workers retry from the last durable update at most twice, then independent experiments continue.
Dependent stages never use incomplete or failed pretraining. No automatic data/batch/epoch reduction.
Check pipeline.json, each status.json/console.log, TensorBoard and checkpoint timestamps.
A service stop requests a checkpoint at the next completed update; external power/storage failures remain possible.

Legacy single-sweep results are historical controls, NOT a sweep-only control for these new objectives.
Image distillation, camera unfreezing, projection heads, fixed recent target support and training budget differ from legacy runs.
Radar-cell motion reconstruction remains a training auxiliary, not evidence of independent vehicle-motion accuracy.
Depth-anchor errors use radar measurements, not independent dense depth ground truth.
The DAV2 term supervises relative inverse-depth shape; radar provides metric scale.
Five/ten-sweep pretraining keeps the teacher radar anchors at age <= 0.1 seconds in both runs.
All accumulated sweeps still enter VoxelNet; the DINO target support is not expanded in the ten-sweep treatment.
One epoch of a linear readout may underfit. Weak probe IoU does not by itself prove that representation learning failed.
