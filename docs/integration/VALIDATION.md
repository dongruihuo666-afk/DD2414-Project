# Integration cleanup validation

Recorded 13 November 2026 on the workstation, using an isolated source worktree.
No cluster login/submission, job cancellation, data deletion, full retraining,
CUDA workload or existing-environment upgrade was performed for this cleanup.

## Baseline and final checks

| Check actually executed | Result |
| --- | --- |
| Existing pre-extraction CPU suite | 103 passed |
| Same suite after package extraction | 103 passed |
| Named interfaces/parser plus official protocol checks | 111 existing + 8 new tests passed |
| Expanded curated CPU suite including mocked site recovery/control checks | 194 passed, 9.625 s |
| Fixed original/package inputs in separate CPU processes | All ten groups exactly equal, including tensor/gradient hashes |
| Actual official RGB checkpoint, full Segnet `strict=True` | 677 state items, zero missing/unexpected keys |
| Same checkpoint, strict encoder-only initialization | 568 tensors, pinned SHA-256 matches |
| Editable install, wheel build/install and import outside checkout | Passed in temporary venvs; no live environment changes |
| New train/evaluate help and legacy train/DINO/motion/rotation help | Passed, no weight download or training |
| Both portable experiment argument lists | Pass the existing strict random-profile validator with two local ranks |
| Publication command planner | Prints only; no Git publisher invoked |
| Asset verifier | Frozen manifest bytes and BEVCar voxel source hashes match |
| Source syntax/import boundary and shell template syntax | Passed |
| Local Markdown links, including moved archives | All targets valid in 47 checked documents |
| Original core/data/manifest/protocol file byte comparison | All 38 selected tracked files unchanged from `0f91e22` |

Three original multiprocessing checks fail inside a restricted sandbox because
loopback sockets/shared memory are denied. The identical suite passes with
local IPC/socket permission; no model/test assertion was disabled. The expanded
runner initially exposed a child-process import-path issue in the new help
test, corrected by giving that subprocess its explicit source path; the final
194-test run passed. This is distinct from the wheel installation/import check.

The local environment has Python 3.10.14, torch 2.5.1 (CUDA 12.4 build),
torchvision 0.20.1, NumPy 1.23.5, Pillow 9.4.0 and nuScenes SDK 1.1.11.
The CUDA 12.1 mamba template copies existing cluster pins; no new mamba solve
was run. Hosted CI was added but has not been represented as an observed pass.

## Commands

```bash
python tools/quality/check_source.py
python tools/quality/check_links.py
python tools/testing/run_cpu_regressions.py
python -m pip install --no-deps --no-build-isolation -e .
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dd2414-wheel .
dd2414 train --help
dd2414 evaluate --help
bash -n configs/deployment/slurm.example.sh
python tools/verify_asset.py --asset full_data_manifest --path configs/radar_scaling_manifest_seed125.json
```

For the original/package parity commands, see
[compatibility](COMPATIBILITY.md). The ten groups cover semantic and motion
losses/gradients, connected empty-motion gradient, toy-student wrapper state
keys/values/gradients, streamed depth projection, compensated radar rotation,
calibrated input/voxel preparation, acknowledged sampling, LR schedule and the
next optimizer step after checkpoint restore. This is not full-model CUDA or
full-nuScenes prediction parity. Real-data/CUDA integration remains opt-in.

The manifest's **raw-file** SHA-256 is
`f46868da2d68fe651cddce64013981454f67c8574a1a6f46a6723f72bc84002f`;
its preserved **canonical unsigned-JSON** digest is
`dc39006dffbf036d5ea7728dad9348b822c78cd47cd02c226b638271c3420c5d`.
Its content and ordering were not regenerated. Whole-object Python pickle
compatibility is not promised; tensor/state checkpoints retain their schema.

## Old paths and new responsibilities

| Legacy path | Formal implementation / current role |
| --- | --- |
| `scripts/fullsize_training.py` | `dd2414.runtime.fullsize`, alias retained |
| `scripts/baseline_followup.py` | `dd2414.evaluation.followup`, alias retained |
| `scripts/multisweep_teacher.py` | `dd2414.models.multisweep`, alias retained |
| `scripts/metric_depth_bev.py` | `dd2414.models.metric_bev`, alias retained |
| `scripts/metric_bev_objective.py` | `dd2414.objectives.metric_bev`, alias retained |
| `scripts/metric_bev_profile.py` | `dd2414.experiments.metric_profile`, alias retained |
| `scripts/depth_pretraining_objective.py` | `dd2414.objectives.depth`, alias retained |
| `scripts/training_checkpoint.py`, `distributed_training.py` | `dd2414.runtime.checkpoints`, `distributed`, aliases retained |
| `scripts/radar_scaling_data.py`, `radar_evaluation.py` | `dd2414.data.manifest`, `evaluation.radar_controls`, aliases retained |
| `scripts/dinov2_bev_demo.py` | Shared functions in `dd2414.teachers.dino`; original CLI kept |
| `scripts/semantic_distill_smoke.py` | Class/loss in `models.distillation`, `losses.semantic`; original experiment kept |
| `scripts/compare_motion_target_bevcar_tiny.py` | Shared model/input/motion functions in `models.legacy_motion`, `objectives.radar_motion`; experiment kept |
| `scripts/train_fullsize_linear_probe.py` | Generic readout helpers in `evaluation.probe`; original CLI kept |
| `scripts/train_radar_scaling.py` | Generic log/validation helpers in `evaluation.scaling`; original CLI kept |
| `scripts/train_multisweep_teacher.py` | Compatible research driver; same flags parsed by `dd2414.cli.train_args` |
| `scripts/test_radar_velocity_rotation.py` | Wrapper to `tools/visualization/radar_velocity_rotation.py`, integration program |
| Long root history/plan documents | `docs/archive/pre-integration/`, original addresses retained as links |
| Radar technical notes | `docs/reference/RADAR_ENCODER_NOTES.md`, root link retained |
| Implicit queue publication | Manual `tools/publish_results.py` command, explicit `--execute` required |

Selected `artifacts/` files remain at their established paths to preserve report
reproduction/data references. Bulk runtime outputs remain excluded. The
original root checkout remains on `dd2414-mini-baseline` with its staged PDF;
this integration worktree has separate commits/index and never modifies that
staging area. Source/data exclusions and external archives are in [backup](BACKUP.md).
