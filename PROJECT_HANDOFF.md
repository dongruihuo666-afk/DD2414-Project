# DD2414 project technical handoff

This is the short, continuously updated entry point for teammates and coding
agents. `README.md` introduces the repository; `PROJECT_PROGRESS.md` contains
the broader experiment report; this file records the current implementation,
reproduction method, evidence, limitations, and handoff decisions.

## Current state

The official Simple-BEV camera-only and camera-plus-radar pipelines run on
nuScenes v1.0-mini. A separate, opt-in radar point encoder has been integrated
with the camera BEV feature. Frozen DINOv2 targets have driven a four-sample
label-free student experiment using the **legacy radar path**. The new radar
encoder has passed integration tests but has **not** been trained against the
DINO target. A separate adapter and the official, randomly initialized BEVCar
radar encoder have passed a one-frame input/forward/backward CUDA smoke test;
that encoder has **not** been fused with cameras or trained. No motion head or
full-dataset self-supervised evaluation exists.

The collaboration branch is `dd2414-mini-baseline`. Each completed,
project-scoped change is pushed there so teammates can follow the work. Check
the GitHub PR state before assuming that its commits are on `main`. Merge only
after the bounded feature and its checks are complete and the team has reviewed
the PR; the routine per-task push does not merge anything.

## Data and pretrained weights

| Component | Exact choice and provenance | Local role |
| --- | --- | --- |
| nuScenes | Official `v1.0-mini` archive from [nuScenes](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html), not a "mini-batch" | 10 scenes / 404 key samples; this loader exposes 323 train and 81 validation samples |
| Simple-BEV | Official upstream [aharley/simple_bev](https://github.com/aharley/simple_bev) base commit `be46f0ef71960c233341852f3d9bc3677558ab6d` | Official camera and camera+radar checkpoints for baseline inference and supervised sanity checks |
| DINOv2 teacher | Meta's pretrained [`dinov2_vits14`](https://github.com/facebookresearch/dinov2#pretrained-models) ViT-S/14 backbone, **without registers or a task head**; weight file [`dinov2_vits14_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | Extracts 384-channel patch features; `eval()` and `requires_grad_(False)` keep every teacher parameter frozen |
| New radar point encoder | `nets/radar_encoder.py`, randomly initialized in the integration smoke test | Converts seven numeric radar fields to a `(B,64,200,200)` BEV feature |

The DINO code loads from a local PyTorch Hub cache if present, otherwise from
`facebookresearch/dinov2`. **We did not fine-tune DINOv2 on nuScenes.** The
adaptation is geometric: camera calibration and measured radar range place
frozen image features into soft BEV target regions. The trainable student is
Simple-BEV plus a semantic projection head. In the four-sample experiment, the
student starts without a supervised BEV checkpoint, but its original radar
input path is enabled. Distinguish this from the later *new* radar encoder.

Datasets, pretrained weights, official checkpoints, generated student
checkpoints, logs, and raw teacher caches are local dependencies excluded from
Git; teammates must obtain them separately. See `SETUP_LOCAL.md`.

## Implemented pipeline and evidence

```text
six cameras -> calibrated camera BEV -------------------------+
                                                          fusion -> shared BEV
radar points -> new 7-field encoder -> 64-channel radar BEV --+

separate label-free experiment:
six cameras -> frozen DINOv2 patch features --+
radar range + calibration --------------------+-> soft BEV targets
legacy-radar Simple-BEV student ----------------> cosine feature loss
```

| Check | Reproduce | Observed result | Interpretation |
| --- | --- | --- | --- |
| Official camera/radar demo | `./scripts/run_meeting_demo.sh` | One mini frame displayed with both predictions and radar positions | Visual coordinate sanity check, not a benchmark |
| Official fixed subset | `./scripts/run_mini_subset_eval.sh` | Mean IoU 0.121 camera vs. 0.291 camera+radar on 10 fixed validation samples; radar higher on 10/10 | Small-subset official-checkpoint comparison; checkpoints were separately trained |
| Supervised loop | `./scripts/run_baseline_overfit.sh` | Four training-sample IoU 0.273 -> 0.803 | Uses human-box-derived labels; overfit check only |
| DINO target geometry | `./scripts/run_dinov2_bev_demo.sh` | Frozen teacher output `(6,384,14,24)`; radar-anchored local BEV targets | No box-derived target/loss |
| Label-free student | `./scripts/run_dinov2_mini4.sh` | Four-sample mean DINO cosine loss 1.023 -> 0.196 over 60 updates | Trains the legacy-radar student, **not** the new radar encoder; memorization only |
| New radar encoder | `./scripts/run_radar_point_encoder_test.sh` | 403 returns -> 125 quality/ROI points -> `(1,64,200,200)` | Point transform, pooling, empty input, order invariance, and gradient checks pass |
| New fusion integration | `./scripts/run_radar_fusion_test.sh` | Fused `(1,128,200,200)`; gradient norm 0.031452; empty radar and checkpoint reload pass; peak 3.440 GiB | Random-weight engineering check; no accuracy claim |
| BEVCar voxel input | `bash scripts/run_bevcar_voxel_adapter_test.sh` | 403 returns -> 251 in range -> 243 voxels; optional quality filter gives 125 points/123 voxels; zero BEV-cell mismatches | Upstream seventh feature verified as valid-point mask; pretrained-checkpoint compatibility not established |
| Official BEVCar radar encoder | `bash scripts/run_bevcar_encoder_smoke.sh --device cuda` with external BEVCar checkout | 403 returns -> 251 in range -> 243 voxels -> `(1,128,200,200)`; finite nonzero gradients; peak allocated CUDA 0.729 GiB | Isolated random-weight encoder only; no camera fusion, training, accuracy, or checkpoint compatibility claim |

Selected visual results are under `artifacts/`. `PROJECT_PROGRESS.md` explains
each plot and its caveats. The official legacy radar checkpoint still loads
after the new opt-in path was added.

## Reproduction notes

1. Follow `SETUP_LOCAL.md` to install the `simplebev` environment, obtain
   nuScenes mini, and obtain the official Simple-BEV checkpoints. Shell scripts
   accept `CONDA_ROOT`, `CONDA_ENV`, `NUSCENES_ROOT`, and `OUTPUT_DIR` overrides.
2. Start with `./scripts/check_environment.sh` and
   `./scripts/smoke_test_mini.sh --data-only`.
3. Run the bounded scripts in the table above for the specific pipeline under
   investigation. They are not a single end-to-end benchmark command.
4. Check `git status`, branch/PR state, and test outputs before reporting a
   result. Small-sample training curves must not be described as validation.

## Agent handoff workflow

`AGENTS.md` gives repository-wide instructions to Codex. The project-local
`.codex/hooks.json` uses a read-only `Stop` hook to notice uncommitted local
changes **or committed work ahead of the upstream feature branch** and ask for
one final handoff/push review. On each machine, trust the
project's `.codex/` configuration and review the hook with Codex `/hooks`;
untrusted hooks are skipped. The hook never performs a Git write or contacts
GitHub. It remains quiet when the worktree is clean and there are no unpushed
commits. It does not turn a read-only question into permission to edit or push.

After a material code or experiment change, the responsible agent should:

1. update this file's `Current state` and append one dated work-log entry;
2. list the exact changed paths, test command and observed result, limitations,
   and next action;
3. inspect `git status`, stage only in-scope files, commit with an English
   message, push the collaboration branch as the default completion step, and
   verify the remote branch head; and
4. give teammates the commit/PR link and any blocked or omitted step.

The agent should do this **before** declaring the task complete; a lifecycle
hook is a fallback reminder, not a guarantee that interrupted or offline work
can be published.

Routine pushes to the feature branch are already authorized by the project
owner; another conversational confirmation is unnecessary. Codex may still
need to follow an environment-level network or filesystem approval. This
workflow does not authorize automatic `main` merges, force-pushes, or uploading
unrelated local work.

## Next bounded task

Keep the lightweight encoder unchanged. Compare the lightweight and BEVCar
radar paths under the already implemented frozen-DINOv2 target using the same
examples and update budget. Resolve their feature/frame and memory differences
before interpreting losses. Do not use BEVCar's supervised checkpoint for a
label-free claim, and do not add a motion head until this experiment is stable.

## Work log and update template

Append a dated entry here for each material repository-changing task. Keep the
entry concise and factual: changed paths, reason, exact command(s), observed
result, limitation, next action, commit/PR link or push blocker. Update the
`Current state` section when a milestone changes. Do not duplicate entire
chat transcripts, secrets, or unreviewed generated data.

### 2026-09-18 — Official BEVCar radar encoder isolated smoke test

- Change: corrected the seven-field BEVCar adapter after inspecting official
  source commit `29cacda3bc5416d47428c1d0f017527acad34f90` (seventh field
  is a valid-point mask, not time lag); added an opt-in external-source encoder
  smoke script and documentation. Changed `.gitignore`,
  `nets/bevcar_voxel_adapter.py`, `scripts/test_bevcar_voxel_adapter.py`,
  `scripts/test_bevcar_encoder_smoke.py`,
  `scripts/run_bevcar_encoder_smoke.sh`,
  `artifacts/bevcar_voxel_adapter_audit.png`, `README.md`,
  `RADAR_ENCODER_NOTES.md`, and this handoff. The lightweight path is unchanged.
- Verification: `bash scripts/run_bevcar_voxel_adapter_test.sh` and
  `BEVCAR_SOURCE_DIR=<official-checkout> bash scripts/run_bevcar_encoder_smoke.sh
  --device cpu|cuda` on mini token `cd9964f8c3d34383b16e9c2997de1ed0`:
  403 returns, 251 in range, 243 voxels, zero BEV-cell mismatches, output
  `(1,128,200,200)`, finite nonzero SVFE/CML gradients. Isolated CUDA peak
  PyTorch allocation was 0.729 GiB. See the radar notes for exact setup.
- Limitations: one frame, one radar sweep, random weights; no camera fusion,
  trained accuracy, or label-free objective. Our camera-frame velocity correction,
  deterministic point selection and dynamic voxel count differ from BEVCar's
  released preprocessing. Official code remains an external dependency.
- Next: compare both radar branches under identical frozen-DINOv2 targets,
  examples, and update budgets; avoid motion head for now.
- Collaboration: publish on `dd2414-mini-baseline` and continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1), without
  merging `main` as part of this smoke test.

### 2026-09-18 — BEVCar-shaped voxel input geometry audit

- Change: added `nets/bevcar_voxel_adapter.py`,
  `scripts/test_bevcar_voxel_adapter.py`,
  `scripts/run_bevcar_voxel_adapter_test.sh`, and
  `artifacts/bevcar_voxel_adapter_audit.png`; updated `.gitignore`, `README.md`,
  `RADAR_ENCODER_NOTES.md`, and this handoff. The existing lightweight radar
  encoder and model paths were not changed.
- Reason: establish a shared camera-reference BEV coordinate grid and the
  BEVCar VoxelNet input tensor shapes before considering network transplant.
- Verification: `bash scripts/run_bevcar_voxel_adapter_test.sh` passed on the
  first fixed mini training frame (token `cd9964f8c3d34383b16e9c2997de1ed0`):
  403 returns, 251 in-range, 125 quality-filtered, 123 3D voxels, no point
  truncation, zero BEV-cell mismatches; synthetic collision, empty/batched,
  overflow, and axis checks passed. `git diff --check` passed.
- Limitation: this is CPU input/coordinate validation, not BEVCar encoder
  inference, backward, GPU memory, training, or proof of pretrained-checkpoint
  compatibility. Its explicit seventh feature and deterministic overflow
  policy need upstream preprocessing comparison. The current environment
  denied NVIDIA access, so the GPU-only `VizData` loader was bypassed using
  the same first nuScenes key frame and direct calibrated-sensor transforms.
- Next: confirm upstream feature order and point sampling, then add a separate
  random-weight BEVCar encoder smoke test without replacing the light baseline.
- Collaboration: pushed through the `dd2414-mini-baseline` feature branch and
  tracked in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  this entry does not imply a merge into `main`.

### 2026-09-18 — Push-on-feature-branch workflow

- Change: clarified that each completed project change is committed and pushed
  to the active feature branch without a new conversational permission request;
  the `Stop` reminder now also detects commits ahead of the upstream branch.
- Reason: keep teammates' clones and coding agents synchronized while reserving
  `main` for reviewed, completed features.
- Verification: hook JSON, Python compile, simulated dirty/continued events,
  and a real ahead-of-upstream check before the push of this documentation
  change.
- Limitation: the hook itself never pushes and cannot bypass Codex environment
  approvals or guarantee publication while offline.
- Next: finish and review the radar+DINO experiment before deciding when to
  merge its feature branch into `main`.

### 2026-09-18 — Technical handoff and Codex workflow

- Change: documented dataset/DINO checkpoint provenance, frozen-teacher versus
  trainable-student roles, reproducible experiments, and current radar-fusion
  status; added repository-level agent instructions and a read-only `Stop` hook
  that reminds Codex to review unfinished local changes.
- Reason: let teammates and their coding agents distinguish completed results
  from the next experiment and find the exact reproduction commands.
- Verification: `python3 -m json.tool .codex/hooks.json`, Python compile,
  simulated dirty/continued `Stop` events, Markdown link inspection, and
  `git diff --check` passed. The simulated dirty event requested one handoff
  continuation; a second event with `stop_hook_active=true` returned `{}`.
- Limitation: hooks must be trusted in each Codex installation and cannot
  safely auto-push arbitrary local changes; read-only chats should not mutate
  the repository.
- Next: run the new radar+DINO tiny-subset experiment after team review.
