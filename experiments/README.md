# Research and integration program index

Reusable code lives in `src/dd2414`. This index keeps historical commands
discoverable without copying or deleting them because of a tiny/demo/smoke name.

| Purpose | Preserved command |
| --- | --- |
| Teacher/target visualization | `scripts/dinov2_bev_demo.py` |
| Small semantic distillation experiment | `scripts/semantic_distill_smoke.py` |
| Historical motion comparison | `scripts/compare_motion_target_bevcar_tiny.py` |
| Held-out image/BEV comparison | `scripts/eval_image_distill_heldout.py` |
| Radar/voxel real-data admission | `scripts/test_bevcar_voxel_adapter.py`, `scripts/test_bevcar_encoder_smoke.py` |
| Historical radar coordinate visualization | `tools/visualization/radar_velocity_rotation.py` (old script retained as wrapper) |
| Radar feature experiment | `scripts/test_radar_point_encoder.py` |
| Small-data original model integration | `scripts/smoke_test_mini.py` |
| Completed official full-data matrix | `scripts/run_official_full_data.py` (archived study controller) |

These commands are opt-in and may require actual data, external source/weights,
CUDA or write plots/results. Do not run them through blanket pytest discovery.
Current full-data protocols are indexed in [docs](../docs/README.md). Subset
records remain scientific history; they are not the recommended training path.
Old study controllers may enforce an original branch/source/budget on purpose.
Use their tagged source for original-source recovery, not a renamed new run.
