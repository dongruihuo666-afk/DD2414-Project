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
encoder has passed camera-fusion integration tests but has **not** been trained
against the DINO target in that fused model. A separate adapter and official
BEVCar encoder passed a one-frame CUDA smoke test. Both radar encoders have
now been optimized **standalone** on the same four cached frozen-DINOv2
targets. A standalone 12-frame, two-scene validation check with three seeds
reduced target loss, but a cross-scene radar-swap control found little or no
radar-alignment effect. Further no-radar and measurement ablations found that
the target loss does not require radar measurements in either branch and does
not require radar at all for the BEVCar branch. The encoders are now compared under a matched supervised camera-fusion
budget; the larger BEVCar encoder shows a clearer radar-dependence gap
(see the 2026-09-28 work log entry). A separate **supervised**
Simple-BEV + BEVCar mini demo now shows a modest radar-dependent IoU gain on
adaptation-held-out frames using box-derived labels; it does not solve the
self-supervised objective. A switchable **camera-side dense semantic
distillation** (`--camera-distill` in `semantic_distill_smoke.py`) now distills
the camera-only BEV against the dense camera-visibility DINO target alongside
the existing radar-guided fusion term; on a single-sample smoke it reaches the
representation only with an unfrozen encoder and does not yet beat the
fusion-only baseline (see the 2026-09-28 work log entry).
No motion head or full-dataset self-supervised evaluation exists.

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
| Radar-only frozen-DINOv2 comparison | `bash scripts/run_radar_dino_tiny.sh --samples 4 --steps 60` with cached targets and external BEVCar | Same-frame cosine loss: light 0.986 -> 0.890, BEVCar 1.002 -> 0.376; finite encoder gradients | Four training frames only; different capacity/filtering; not held-out accuracy or camera-radar fusion |
| Radar-only mini validation | `bash scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12 --seed-list 125,126,127` | 12 unseen frames, 2 scenes, 3 seeds: loss light 0.979 -> 0.903, BEVCar 0.995 -> 0.493; all 36 seed-frame losses fell | Cross-scene radar swap: light 0.909, BEVCar 0.490, versus aligned 0.903/0.493; no robust BEVCar radar-alignment evidence |
| Fixed-model radar ablation | Add `--diagnose-radar` to the validation command | Correct/no-radar loss light 0.903/0.911; BEVCar 0.493/0.489; zeroing or mixing radar numeric values barely changes either | Current DINO target loss does not prove use of radar measurements; BEVCar needs no radar to retain its low loss |
| Supervised BEVCar fusion demo | `bash scripts/run_bevcar_supervised_mini.sh --steps 80 --seed 125` and repeat seeds 126/127, then `python scripts/summarize_bevcar_supervised.py` | Mean 12-frame IoU across 3 seeds: correct radar 0.180, empty 0.166, wrong scene 0.171; SVFE/CML gradients nonzero | Human-box-derived labels; 4-frame adaptation, camera/decoder pretrained on broader nuScenes; not self-supervised or matched-budget benchmark |

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

Keep the lightweight encoder unchanged. Treat the supervised BEVCar result
as a small positive radar-use control, not a solution to DINOv2 insensitivity.
Investigate why the self-supervised target loss ignores radar removal and
measurements; test output-map sensitivity and add an objective/control in
which matched radar beats empty and mismatched radar. Any architecture claim
requires matched data/update budgets and a larger genuinely unseen split.
Resolve capacity, feature/filter, frame, and memory differences.
Do not use BEVCar's supervised checkpoint for a label-free claim, and do not
add a motion head yet.

## Work log and update template

Append a dated entry here for each material repository-changing task. Keep the
entry concise and factual: changed paths, reason, exact command(s), observed
result, limitation, next action, commit/PR link or push blocker. Update the
`Current state` section when a milestone changes. Do not duplicate entire
chat transcripts, secrets, or unreviewed generated data.

### 2026-09-28 — Matched lightweight-vs-BEVCar supervised comparison

- Change: extended `scripts/bevcar_supervised_mini.py` with an opt-in
  `--encoder {bevcar,light}` flag so the lightweight 7-field point encoder
  runs under the same supervised mini protocol as the BEVCar meeting demo.
  The BEVCar-only gradient checks and the success sentinel are now conditional
  on that flag. Whitelisted `artifacts/light_supervised_*.json` and `.png` in
  `.gitignore`. Updated `RADAR_ENCODER_NOTES.md` and this handoff. Official
  Simple-BEV and lightweight encoder paths are unchanged.
- Reason: the supervisor asked whether a stronger encoder makes radar matter
  more. The existing meeting demo ran only BEVCar and was explicitly not a
  matched-budget comparison; this adds the lightweight branch under an
  identical 4-frame adaptation / 12-frame two-scene validation / three-seed /
  80-update protocol.
- Verification: `bash scripts/run_bevcar_supervised_mini.sh --encoder light
  --steps 80 --seed <125|126|127>` passed for all three seeds, writing
  `artifacts/light_supervised_mini*.json`. Mean 12-frame validation IoU was
  correct radar 0.188, empty radar 0.188, wrong-scene radar 0.188 (light),
  versus the committed BEVCar 0.180 / 0.166 / 0.171. Correct radar beat empty
  in 19/36 light cases (29/36 for BEVCar) and wrong radar in 22/36 (30/36).
  Mean correct-empty probability difference was 0.000054 (light) versus
  0.003379 (BEVCar). Peak allocated CUDA memory 3.618 GiB; about 9.3 s per
  seed. `python -m py_compile` and `git diff --check` passed.
- Limitations: the two branches use different radar input features and
  filtering (lightweight keeps a quality mask and seven numeric fields; the
  BEVCar adapter keeps all in-range returns with seven channels), so encoder
  size is confounded with preprocessing. BEVCar's larger radar-dependence gap
  coexists with a slightly lower absolute correct IoU (0.180 vs 0.188) in this
  4-frame budget. Two mini scenes, three seeds, one radar sweep, box-derived
  labels; no self-supervised or motion objective.
- Next: use this supervised gap as the encoder-change answer and keep the
  self-supervised objective/control redesign as the separate open task.
  Continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1).

### 2026-09-18 — Supervised Simple-BEV + BEVCar meeting demo

- Change: added `nets/bevcar_radar_bridge.py` to import official external
  VoxelNet and project its output into the existing Simple-BEV experimental
  fusion interface. Added `scripts/bevcar_supervised_mini.py`, its shell
  launcher, `scripts/summarize_bevcar_supervised.py`, three-seed JSON metrics,
  a representative prediction/radar/label panel, and a three-seed summary
  figure; updated `.gitignore`, `README.md`, `RADAR_ENCODER_NOTES.md`, and this
  handoff. The official legacy and lightweight paths remain unchanged.
- Reason: separate the earlier self-supervised objective failure from the
  question of whether BEVCar VoxelNet can help under Simple-BEV's original
  human-box-derived labels. Transfer official camera-only encoder/decoder and
  image-fusion weights, freeze camera/decoder, train new radar/fusion layers.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_bevcar_supervised_mini.sh --steps 80 --train-samples 4
  --val-samples 12 --seed <125|126|127>` passed for all three seeds;
  `python scripts/summarize_bevcar_supervised.py` produced the summary. Mean
  12-frame validation IoU was correct radar `0.180`, same fused model with
  empty radar `0.166`, wrong-scene radar `0.171`, and unadapted camera-only
  context `0.121`. Correct radar beat empty in 29/36 seed-frame cases and
  wrong radar in 30/36. SVFE and CML gradient norms were positive in each
  run; peak allocated CUDA memory was 3.623 GiB. A one-step smoke test passed.
- Limitations: supervised labels are derived from human 3D boxes; DINOv2 is
  absent. Only four mini frames were used for adaptation; the 12 validation
  frames are scene-disjoint from that adaptation, but the official camera
  checkpoint was pretrained on broader nuScenes data. The unadapted camera
  model is not a matched-budget comparison. IoU gains are modest and not
  universal across frames. One radar sweep and a custom adapter differ from
  released BEVCar preprocessing; no BEVCar supervised weights were used.
- Next: keep this as the supervised meeting baseline and fix the separate
  self-supervised objective/controls before claiming label-free radar learning.
  Continue review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` as a routine push step.

### 2026-09-18 — Fixed-model radar-input ablation

- Change: extended `scripts/compare_radar_dino_tiny.py` with optional
  `--diagnose-radar` evaluation for empty radar, positions only, permuted
  numeric measurements, and the existing cross-scene swap. Added
  `artifacts/radar_dino_ablation.json` and `.png`; updated `.gitignore`,
  `README.md`, `RADAR_ENCODER_NOTES.md`, and this handoff. Training models,
  teacher targets, and official Simple-BEV paths were not altered.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12
  --seed-list 125,126,127 --diagnose-radar` passed. Across 36 seed-frame
  evaluations, mean correct/no-radar loss was light `0.903/0.911` and BEVCar
  `0.493/0.489`; positions-only `0.903/0.495`; permuted-measurements
  `0.903/0.493`. The preliminary two-frame smoke test also passed.
- Additional checks: synthetic CPU inputs confirmed positions/masks remain
  unchanged in the intended ablations, numeric values are permuted rather
  than dropped, and empty BEVCar input has zero occupied voxels. Python
  compile and `git diff --check` passed.
- Interpretation: the lightweight branch has a small radar-geometry effect,
  but numeric measurements barely affect this target loss. BEVCar's low
  target loss does not require any radar under these interventions. This is
  evidence about the objective/metric, not proof that the architecture is
  incapable of radar perception or that its outputs are literally identical.
- Limitation: two mini validation scenes and three seeds, with unequal model
  capacity/filtering. Frozen-DINOv2 cosine loss is not detection accuracy;
  output-map sensitivity and a radar-dependent objective remain untested.
- Next: inspect output-map changes and test an objective/control requiring
  correctly paired radar to outperform empty/mismatched radar. Continue
  review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` based on loss reduction alone.

### 2026-09-18 — Two-scene, three-seed held-out radar diagnostic

- Change: extended `scripts/compare_radar_dino_tiny.py` with frozen-DINOv2
  target generation on 12 uniformly spaced mini validation frames, three
  initialization seeds, scene/token separation checks, and a cross-scene
  radar-swap control. Added `artifacts/radar_dino_heldout_comparison.json`
  and `.png`; updated `scripts/dinov2_bev_demo.py` with an optional validation
  loader selection, `.gitignore`, `README.md`, `RADAR_ENCODER_NOTES.md`, and
  this handoff. Existing Simple-BEV and lightweight paths remain unchanged.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60 --heldout-samples 12
  --seed-list 125,126,127` passed. Six frames per validation scene; all 36
  seed-frame losses fell for both branches. Mean loss light `0.979 -> 0.903`,
  BEVCar `0.995 -> 0.493`. Final seed-mean ranges were `0.864-0.925` and
  `0.475-0.515`. The paired cross-scene radar swap yielded `0.909` (light)
  and `0.490` (BEVCar), versus aligned `0.903` and `0.493`.
- Limitation: BEVCar's validation loss reduction survives unseen scenes but
  does not show positive dependence on frame-aligned radar under this control.
  The light branch shows only a small alignment effect. Frozen-teacher cosine
  loss is not downstream segmentation accuracy, and only two validation
  scenes were available. Capacity/filtering are still unequal; no complete
  camera-radar fusion or motion model was tested.
- Next: strengthen radar-dependence tests before choosing an encoder or
  claiming radar semantic learning. Continue review in
  [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  no automatic `main` merge.

### 2026-09-18 — Four-frame radar-only DINOv2 target comparison

- Change: added `scripts/compare_radar_dino_tiny.py` and its shell entry,
  `artifacts/radar_dino_tiny_comparison.json` and `.png`; updated `.gitignore`,
  `README.md`, `RADAR_ENCODER_NOTES.md`, and this handoff. Existing model paths
  remain untouched.
- Reason: test whether the lightweight and official BEVCar radar encoders can
  each optimize the same cached frozen-DINOv2 BEV targets on four mini frames,
  using 60 updates per branch and no human box-derived loss.
- Verification: `BEVCAR_SOURCE_DIR=<official-checkout> bash
  scripts/run_radar_dino_tiny.sh --samples 4 --steps 60` passed. Same-frame
  mean cosine loss was light `0.985960 -> 0.889696` and BEVCar
  `1.001846 -> 0.375588`; encoder gradients were finite/nonzero. Peak allocated
  CUDA was 1.070/1.009 GiB respectively. Calibration matched the target cache.
- Limitations: radar-only and all four frames were training samples. BEVCar
  has 495,328 trainable parameters versus 35,712 in the light student, and
  they differ in filtering/features; the loss is not held-out accuracy.
  No camera fusion, supervised checkpoint, or motion model was involved.
- Next: held-out mini data, then camera-radar integration under a fixed
  protocol. Continue review in [PR #1](https://github.com/dongruihuo666-afk/DD2414-Project/pull/1);
  do not merge `main` merely because this optimization check passed.

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

### 2026-09-28 — Camera-side dense semantic distillation switch (opt-in ablation)

- Change: added a switchable camera-side DINOv2 semantic distillation term so the
  dual-level distillation from the meeting can be compared against the existing
  fusion-only term. `nets/segnet.py` now returns the pure camera BEV
  (`camera_bev`, B, feat2d_dim*Y, Z, X) alongside the fused `feat_bev` when
  `return_shared_bev=True`; `scripts/dinov2_bev_demo.py` additionally caches
  `dense_semantic_target` (the naive-rays `teacher_bev`) and `camera_coverage`
  (camera visibility) alongside the existing radar-anchored `soft_targets`;
  `scripts/semantic_distill_smoke.py` gains `--camera-distill`,
  `--camera-weight`, and a decoupled `--train-encoder` flag plus a small
  `camera_head` that maps the camera BEV to the DINOv2 feature space.
- Reason: the supervisor's dual-level distillation keeps DINOv2 semantics from
  being washed out by radar-guided sparsity. The camera-side term distills the
  DENSE (un-radar-guided) DINO features into the camera BEV before fusion,
  while the existing fusion term keeps the radar-anchored (sparse) targets.
  The only differences between the two terms are the student representation
  (camera-only vs fused) and the target mask (camera visibility vs radar soft
  region).
- Verification: `python3 -m py_compile` on the three changed files and
  `git diff --check` passed. Four 20-step single-sample smoke configs
  (cosine-to-DINO loss, NOT a validation result) gave fusion losses:
  baseline (frozen, camera off) 0.2039; +camera-distill (frozen) 0.2039;
  +train-encoder only (trainable, camera off) 0.1730; +train-encoder
  +camera-distill (trainable) 0.1767. Peak CUDA ~4.2 GiB.
- Limitation: the camera term reaches the representation only when the encoder
  is unfrozen, and the fusion-loss gain is driven by encoder trainability
  (0.1730) rather than the dense camera term (0.1767, slightly worse) —
  consistent with the naive-rays depth ambiguity of `teacher_bev`. Single mini
  sample, 20 updates; these are training losses, not held-out results.
- Next: replace the dense target with a depth-disambiguated one (e.g. lift the
  camera BEV to the radar-anchored depth) or evaluate on a held-out split
  before concluding on the camera term.
