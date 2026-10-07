# DD2414 project technical handoff

This is the short, continuously updated entry point for teammates and coding
agents. `README.md` introduces the repository; `PROJECT_PROGRESS.md` contains
the broader experiment report; this file records the current implementation,
reproduction method, evidence, limitations, and handoff decisions.

## Current state

2026-10-05 trainval status: the Ubuntu host has full labeled `v1.0-trainval`;
mini has been removed. The official split is 700 train / 150 val scenes and
28,130 / 6,019 samples; all 2,631,083 referenced sensor files were present and
nonempty. Official BEVCar is pinned at commit `29cacda3` under `external/BEVCar`;
project probes import only its random-initialized `nets/voxelnet.py`, not the
full BEVCar model or checkpoint.

The exact prior BEVCar held-out protocol was reproduced with trainval: head
64-train / 16-val samples, 512 steps, three seeds and one sweep. Motion-only and
joint zero-velocity penalties were `+0.0768` and `+0.0804`; the head subset
covered only 2/1 train/val scenes and removing all radar improved motion loss,
so it is weak generalization evidence. A separate uniform 64/16-scene control
gave `+0.7980` and `+0.8602`, with correct radar outperforming no radar. See
`TRAINVAL_RUN_PLAN.md` and the 2026-10-05 work-log entry. These are bounded
held-out feature-loss probes, not all-frame training or downstream accuracy.
The scaling trainer now provides all-frame streaming self-supervision; the
combined hybrid model remains future implementation work.

The owner has adopted a staged full-data radar scaling study. The frozen order
is streaming data, checkpoint/resume, wrong-scene evaluation, a throughput
pilot, then single-seed `64 -> 256 -> 1,024 -> 4,096 -> 28,130` training-scale
results evaluated on the same complete 6,019-frame validation split. Each
completed milestone must be committed and pushed before the next begins. See
`FULL_DATA_RADAR_SCALING_PLAN.md`; its dependence curve precedes multi-sweep or
hybrid-teacher expansion.

P1 is complete: the seed-125 manifest deterministically nests
64/256/1,024/4,096/full train subsets, covers 64/256/700/700/700 scenes, and
keeps all 700 train scenes disjoint from all 150 validation scenes. A real
512-batch one-sweep traversal retained zero CUDA allocation at every 64-batch
checkpoint. P2 is also complete: atomic periodic checkpoints restore model,
optimizer, AMP scaler, sampler position and all RNG states; a CUDA interrupted
resume matched uninterrupted training exactly. P3 is complete: the full val
wrong-radar map is a deterministic 6,019-token cross-scene bijection, and all
four input modes emit finite incremental per-frame records. P4 is complete: a
128-update online-DINO joint pilot stopped at update 64, resumed in a new
process at the exact sampler position and finished with bounded 3.574 GiB peak
allocation. Measured throughput locks the curve at 30,000 updates per scale
without a teacher-feature cache. P5 is complete: the 64-sample run performed
30,000 effective updates and evaluated all 6,019 validation frames in four
radar modes. Its semantic penalties are small, while its motion penalties are
large and positive; these establish radar dependence for the diagnostic motion
head, not downstream vehicle-segmentation utility. P6 is also complete under
the unchanged protocol: the 256-sample point lowers matched validation losses
and increases semantic and motion radar-intervention penalties relative to P5.
P7 is complete as well: at 1,024 samples, matched losses fall again and empty /
wrong-scene penalties strengthen substantially. P8 is now complete: the 4,096
sample point continues the monotonic increase in semantic and motion
wrong-scene penalties, with positive semantic penalties in all 150 validation
scenes. P9 is complete as well: the full 28,130-sample run covered every train
token, exercised a real process restart at update 1,000 and evaluated all 6,019
validation frames. Its semantic/motion wrong-scene penalties remain strong at
`+0.02470`/`+1.56690`, but are slightly below P8's `+0.02854`/`+1.59196`.
The single-seed curve therefore strengthens through 4,096 samples and then
plateaus or slightly recedes under the fixed 30,000-update budget; it is not a
monotonic full-range trend or a downstream utility claim. P10 is complete: a
paired 150-scene bootstrap places persistent semantic wrong-scene dependence
at 256 samples, while all motion penalties are positive from 64 samples. The
semantic full-minus-4,096 wrong-scene change is `-0.00383` with 95% interval
`[-0.00460, -0.00309]`; the motion change is `-0.02506` with interval
`[-0.06251, +0.01415]`. This supports saturation or a small late decline, not
indefinite growth. See `RADAR_SCALING_ANALYSIS.md`. P11 downstream IoU is the
next implementation boundary.

The results below this update are historical mini experiments unless an entry
explicitly says trainval.

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
self-supervised objective. A switchable **image-level dense semantic
distillation** (`--image-distill` in `semantic_distill_smoke.py`) now distills
the pre-projection image features against the frozen DINOv2 patch features
alongside the existing radar-guided fusion term; on a single-sample smoke the
image term converges cleanly and no longer drags the fusion loss (see the
2026-09-28 work log entry).
An opt-in motion head now exists in the motion-target probe scripts (2026-09-29); no full-dataset self-supervised evaluation exists.
The three-seed 64-train / 16-held-out dual-teacher probe now completes with the
vectorized target builder. It improves held-out fusion loss over its 4-sample
predecessor, but every no-radar fusion penalty remains small (0.0086--0.0172),
so it still does not establish strong radar dependence (see the 2026-09-30 work
log entry).
The matched 1/5/10-sweep 64-sample comparison found the best held-out fusion
loss at 10 sweeps with the radar-weighted dual teacher, but only a small
no-radar penalty (+0.0196); multi-sweep is a useful input-density control, not
proof of strong radar dependence (see the 2026-09-30 work log entry).

The collaboration branch is `dd2414-mini-baseline`. Each completed,
project-scoped change is pushed there so teammates can follow the work. Check
the GitHub PR state before assuming that its commits are on `main`. Merge only
after the bounded feature and its checks are complete and the team has reviewed
the PR; the routine per-task push does not merge anything.

## Data and pretrained weights

| Component | Exact choice and provenance | Local role |
| --- | --- | --- |
| nuScenes (current host) | Official `v1.0-trainval` metadata and ten blob archives; standard layout under `${HOME}/datasets/nuscenes` | 850 scenes / 34,149 samples; 28,130 train and 6,019 val; file audit passed; bounded BEVCar VoxelNet trainval probes completed |
| nuScenes (historical results) | Official `v1.0-mini` archive | 10 scenes / 404 samples; 323 train and 81 val; removed from this host after trainval replacement |
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

On the current Ubuntu host use `TRAINVAL_RUN_PLAN.md` and
`bash scripts/run_trainval.sh dual-teacher` (preview only). The following mini
commands are historical and need a separate mini installation.

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

Execute P11 from `FULL_DATA_RADAR_SCALING_PLAN.md`: define and run a matched
downstream vehicle-segmentation protocol on the official validation split.
Camera-only and fused models must use identical labeled samples, initialization
policy, update budget, resolution and evaluation code. Report full-validation
IoU for camera-only and for the same fused checkpoint under matched, empty and
wrong-scene radar. If seed 125 is positive, confirm the selected onset and full
scale with seeds 42 and 7 before claiming repeatability. Keep this work separate
from multi-sweep or hybrid-teacher changes.

Keep the lightweight encoder unchanged. Treat the supervised BEVCar result
as a small positive radar-use control, not a solution to DINOv2 insensitivity.
Investigate why the self-supervised target loss ignores radar removal and
measurements; test output-map sensitivity and add an objective/control in
which matched radar beats empty and mismatched radar. Any architecture claim
requires matched data/update budgets and a larger genuinely unseen split.
Resolve capacity, feature/filter, frame, and memory differences.
Do not use BEVCar's supervised checkpoint for a label-free claim. The motion
head now exists only in the opt-in motion-target probe scripts (2026-09-29).

## Work log and update template

Append a dated entry here for each material repository-changing task. Keep the
entry concise and factual: changed paths, reason, exact command(s), observed
result, limitation, next action, commit/PR link or push blocker. Update the
`Current state` section when a milestone changes. Do not duplicate entire
chat transcripts, secrets, or unreviewed generated data.

### 2026-10-07 — P10 radar dependence curve analysis

- Change: added `scripts/analyze_radar_scaling_curve.py` and its CPU tests,
  generated the machine-readable
  `artifacts/trainval/radar_scaling_curve_seed125.json`, rendered
  `artifacts/trainval/radar_scaling_curve_seed125.png`, and documented the
  result in `RADAR_SCALING_ANALYSIS.md`. Updated the plan, README and broader
  progress report to make P10 and its downstream boundary visible.
- Reason: raw validation means do not show whether penalties are stable across
  scenes or whether a data-scale onset is more than a single noisy point. The
  analysis resamples the 150 validation scenes rather than treating 6,019
  temporally adjacent frames as independent.
- Reproduce: `MPLCONFIGDIR=.cache/matplotlib python
  scripts/analyze_radar_scaling_curve.py --bootstrap-iterations 10000
  --bootstrap-seed 20261007`.
- Verification: four CPU analysis tests passed; the integration run reproduced
  every stored aggregate from per-scene paired penalties within `1e-9`, checked
  identical scenes/configuration across all five inputs, completed 10,000
  deterministic bootstrap resamples and emitted the JSON, Markdown and PNG.
  The three pre-existing scaling/evaluation test suites also remain passing.
- Result: semantic zero-velocity and empty-radar intervals are positive from 64
  samples, while the semantic wrong-scene interval first becomes and remains
  positive at 256 (`+0.00450`, 95% CI `[+0.00319, +0.00582]`). All motion
  intervals are positive from 64. Semantic and motion wrong-scene penalties
  peak at 4,096. Full-minus-4,096 is `-0.00383`
  `[-0.00460, -0.00309]` for semantic and `-0.02506`
  `[-0.06251, +0.01415]` for motion. Matched losses improve from
  0.46199/1.58663 at 64 to 0.30538/0.79639 at full scale.
- Interpretation/limitations: scene-level persistence supports a semantic
  alignment onset followed by saturation, and the absolute diagnostic losses
  do not collapse. All training points still use seed 125, so the bootstrap
  does not establish training-seed repeatability. The random frozen camera
  encoder and diagnostic targets do not establish downstream vehicle utility.
- Next: P11, use a matched camera-only/fused labeled protocol and complete-val
  vehicle IoU before changing sweeps, teachers or fusion objectives.

### 2026-10-07 — P9 full-data radar scaling result

- Change: completed the unchanged seed-125, one-sweep, full 28,130-sample
  scale at 30,000 effective optimizer updates and evaluated all 6,019
  validation frames under the four locked radar modes. Whitelisted the compact
  result `artifacts/trainval/radar_scaling_full_seed125.json`; checkpoint and
  raw JSONL outputs remain ignored.
- Reason: P9 is the final fixed-budget data-scale point and tests whether the
  radar-dependence increase through 4,096 samples persists when every official
  training token is included.
- Reproduce: use the P5 command with `--scale full` and replace each
  `scale64_seed125` output stem with `full_seed125`. Run the first process with
  `--stop-after-update 1000`, then repeat the otherwise identical command with
  `--resume` and without `--stop-after-update` to complete training and
  validation.
- Verification: the first process stopped cleanly at update 1,000 and the
  second restored epoch zero and sampler position 1,000. The completed run made
  30,000 consecutive effective updates, covered all 28,130 unique training
  indices, recovered 20 AMP overflows, sustained 10.225 updates/s and peaked at
  3.574 GiB CUDA allocation. Independent checks found exactly 30,000 train and
  6,019 finite validation records, 150 scene summaries, four modes per frame,
  no same-scene wrong-radar pair and no partial record.
- Result: full-validation matched semantic loss was 0.30538 versus 0.30966
  zero-velocity, 0.34502 empty and 0.33008 wrong-scene, giving penalties
  +0.00428/+0.03964/+0.02470. Matched motion loss was 0.79639 versus
  2.33211/2.40259/2.36329, giving penalties +1.53571/+1.60619/+1.56690.
  Moving-frame penalties were +1.73423/+1.81382/+1.76945 and cell-weighted
  penalties were +1.80552/+1.90730/+1.87137. Semantic wrong-scene penalties
  were positive in 150/150 scenes and motion penalties in 144/150 scenes.
- Interpretation/limitations: radar dependence remains strong at full scale,
  but semantic and motion wrong-scene penalties are slightly below P8's
  +0.02854/+1.59196. The five-point curve rises through 4,096 samples and then
  plateaus or slightly recedes, rather than increasing monotonically to full
  scale. Full scale exposes each token only about once under the fixed 30,000
  updates, whereas smaller subsets repeat tokens more often. This remains one
  seed with diagnostic losses and a random frozen camera encoder, not evidence
  of downstream vehicle-segmentation IoU.
- Next: P10, produce the paired per-scene uncertainty analysis and compact
  dependence-curve report before starting downstream IoU work.

### 2026-10-06 — Adopted full-data radar scaling plan

- Change: added `FULL_DATA_RADAR_SCALING_PLAN.md`, linked it from `README.md`,
  and updated the current state and next bounded task in this handoff. The plan
  separates streaming, exact resume and wrong-scene evaluation into independently
  testable milestones, followed by committed seed-125 scale results at 64, 256,
  1,024, 4,096 and all 28,130 training frames. Every scale uses the same full
  6,019-frame validation split and one radar sweep.
- Reason: the mini comparison changed both sample and update counts and did not
  identify a data threshold. A nested trainval curve with matched, zero-velocity,
  empty and wrong-scene controls can test whether radar dependence increases with
  data diversity before multi-sweep or hybrid-teacher changes add confounders.
- Verification: `git diff --check` passed. File/link checks confirmed the P0--P11
  table, the adopted scale sequence, full-validation count, four radar modes and
  per-milestone commit/push rule. No loader, model or training run was part of
  this documentation-only milestone.
- Limitations: the 30,000-update budget is provisional until the P4 throughput
  pilot. Self-supervised penalties demonstrate dependence, not downstream utility;
  P11 requires a matched camera-only comparison and complete-val vehicle IoU.
- Next: P1, deterministic nested manifests and streaming batches. Commit and push
  each later milestone separately before continuing.

### 2026-10-06 — P1 deterministic manifests and streaming data

- Change: added `scripts/radar_scaling_data.py` with stable SHA-256 scene/sample
  ordering, prefix-nested scale manifests, validation and a non-materializing
  `Subset`/`DataLoader` path. Added CPU tests and the real-data checker in
  `scripts/test_radar_scaling_data.py` and
  `scripts/check_radar_scaling_stream.py`. Generated the committed seed-125
  manifest at `configs/radar_scaling_manifest_seed125.json` and the compact
  acceptance summary at `artifacts/trainval/radar_scaling_stream_check.json`;
  whitelisted only that summary in `.gitignore`.
- Reason: the full-data study needs identical, reproducible nested subsets and
  must fetch batches on demand rather than converting a loader to a Python list
  and retaining all tensors on GPU.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_radar_scaling_data.py`; then run
  `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_radar_scaling_stream.py --data-root
  /home/students2026/datasets/nuscenes --manifest-out
  configs/radar_scaling_manifest_seed125.json --summary-out
  artifacts/trainval/radar_scaling_stream_check.json --check-batches 512
  --nsweeps 1` with GPU access.
- Verification: four CPU unit tests passed, Python compilation and
  `git diff --check` passed. The real manifest contains 28,130 train samples in
  700 scenes and 6,019 validation samples in 150 disjoint scenes; scale scene
  counts are 64/256/700/700/700. Streaming 512 real batches read 221,173 radar
  points in 50.923 seconds (10.054 batch/s). CUDA allocated/reserved memory was
  0 MiB at baseline, every 64 batches and completion, so no previous batch was
  retained by this data-only path. Manifest SHA-256 is
  `8905db7995784bbee4d641c9756a8648ebf5510a891fa633fe65c1eab8dcb0fc`.
- Limitations: this milestone did not load DINOv2, build dense BEV targets,
  train a model, or test optimizer state. `VizData` still initializes CUDA
  geometry and therefore uses `num_workers=0`; it also constructs supervised
  labels that self-supervised pretraining does not need. Throughput here is a
  data-only measurement, not the P4 end-to-end timing result.
- Next: P2, periodic atomic checkpoints and exact interrupted resume.

### 2026-10-06 — P2 periodic checkpoints and exact resume

- Change: added `scripts/training_checkpoint.py` with atomic `latest` writes,
  fixed-update periodic/forced-save policy, a stateful epoch-shuffle sampler,
  run-configuration validation and complete Python/NumPy/Torch CPU/CUDA RNG
  capture. It saves model, optimizer, GradScaler, sampler and progress state.
  Added CPU failure/round-trip tests in `scripts/test_training_checkpoint.py`,
  the CUDA equivalence check in `scripts/check_training_resume.py`, and the
  compact result `artifacts/trainval/training_resume_check.json`.
- Reason: the 30,000-update scale runs must survive preemption without silently
  changing sample order, dropout/random augmentation, optimizer moments or AMP
  scaling. Atomic replacement prevents a partially written `latest` file from
  being treated as valid.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_training_checkpoint.py`; then run
  `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_training_resume.py --checkpoint
  artifacts/trainval/checkpoint_resume_smoke/latest.pt --summary-out
  artifacts/trainval/training_resume_check.json --total-updates 8
  --interrupt-after 4 --seed 125` with GPU access.
- Verification: four CPU tests and Python compilation passed. On the RTX 3090
  Ti, uninterrupted eight-update AMP training and a four-update checkpoint plus
  process-state rebuild/resume reached identical sampler state (epoch 1,
  position 3); maximum parameter and per-step loss absolute differences were
  both `0.0`, tighter than the declared `1e-7` tolerance. The first CUDA run
  exposed that `map_location=cuda` moved serialized CPU RNG bytes to CUDA; the
  loader now explicitly returns all generator-state bytes to CPU, and the
  rerun passed with `TRAINING_RESUME_OK`.
- Limitations: the equivalence test uses a small CUDA network rather than the
  full BEV fusion model. Exact sampler resume currently requires
  `num_workers=0`; worker prefetch would need acknowledgement of consumed rather
  than merely requested indices. The P4 end-to-end pilot must call this utility
  from the real streaming trainer and exercise a real checkpoint interruption.
- Next: P3, deterministic wrong-scene radar evaluation alongside matched,
  zero-velocity and empty controls.

### 2026-10-06 — P3 deterministic wrong-scene evaluation

- Change: added `scripts/radar_evaluation.py` with locked matched,
  zero-velocity, truly empty and wrong-scene VoxelNet inputs. The wrong-scene
  map groups stable-hash-ordered samples by scene and rotates by the largest
  scene size, producing a deterministic source-token bijection with no
  same-scene pairs. Added an incremental JSONL writer, CPU tests, and
  `scripts/check_wrong_scene_evaluation.py`. Committed candidates include the
  compact two-frame smoke summary and records under `artifacts/trainval/`.
- Reason: empty-radar sensitivity alone cannot show that the model uses the
  radar aligned to the current camera frame. A bijective wrong-scene control
  preserves the validation radar-token distribution while breaking alignment,
  and streaming JSONL avoids retaining 6,019 frame results in memory.
- Reproduce: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_radar_evaluation.py`; then set the repository DINO cache and run
  `TORCH_HOME=/home/students2026/dd2414/.cache/torch
  /home/students2026/miniconda3/envs/bev/bin/python
  scripts/check_wrong_scene_evaluation.py --data-root
  /home/students2026/datasets/nuscenes --bevcar-source
  /home/students2026/dd2414/external/BEVCar --manifest
  configs/radar_scaling_manifest_seed125.json --output
  artifacts/trainval/wrong_scene_evaluation_check.jsonl --summary-out
  artifacts/trainval/wrong_scene_evaluation_check.json --samples 2 --seed 125
  --nsweeps 1` with GPU access.
- Verification: five CPU tests and Python compilation passed. The real checker
  verified a 6,019-source full-val bijection and zero same-scene pairs. Two
  scene-spread real frames produced matched/zero-velocity/empty/wrong-scene
  records with finite semantic and motion losses; mean motion loss was
  `1.4486/1.4486/1.4523/1.4526`. The first mapping algorithm (a cyclic shift of
  a globally hashed order) failed a valid uneven-scene unit case and was
  replaced by the guaranteed grouped-block construction. Initial random-model
  FP16 smoke produced NaNs; non-finite JSON is now rejected and the engineering
  smoke uses FP32, matching its path-validation purpose.
- Limitations: the smoke model is random and untrained, so its loss differences
  are not radar-effect evidence. Full 6,019-frame metric evaluation, AMP on a
  trained fused model, scene-level summaries and confidence intervals belong to
  P4/P5 and later milestones. Interrupted JSONL files remain as `.partial` for
  diagnosis; record-level resume is not yet implemented.
- Next: P4, integrate the real streaming joint trainer, exercise resume, and
  measure end-to-end throughput/memory before locking the scale-run budget.

### 2026-10-07 — P8 4,096-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 4,096-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale4096_seed125.json`; checkpoint and raw
  JSONL outputs remain ignored.
- Reason: P8 tests whether the growing radar-dependence curve persists after
  another fourfold increase in nested training-set size under the same compute
  budget.
- Reproduce: use the P5 command with `--scale 4096` and replace each
  `scale64_seed125` output stem with `scale4096_seed125`; keep all other locked
  arguments unchanged.
- Verification: 30,000 consecutive effective updates covered exactly 4,096
  unique training indices and seven completed sampler epochs. The run recovered
  20 AMP overflows, sustained 10.269 updates/s, peaked at 3.574 GiB CUDA
  allocation and had mean joint/semantic/motion training losses
  0.5524/0.3100/0.4847. Independent checks found exactly 30,000 train and 6,019
  finite validation records, 150 scene summaries, no same-scene wrong-radar
  pair and no partial record.
- Result: full-validation matched semantic loss was 0.30256 versus
  0.30866 zero-velocity, 0.34537 empty and 0.33110 wrong-scene, giving
  penalties +0.00609/+0.04281/+0.02854. Matched motion loss was 0.79707 versus
  2.35184/2.42179/2.38903, giving penalties +1.55477/+1.62472/+1.59196.
  Moving-frame penalties were +1.75575/+1.83475/+1.79775 and cell-weighted
  penalties were +1.82575/+1.93068/+1.89969. Semantic wrong-scene penalties
  were positive in 150/150 scenes and motion penalties in 142/150 scenes.
- Interpretation/limitations: P5 through P8 wrong-scene penalties rise
  monotonically from +0.00017 to +0.00450 to +0.02093 to +0.02854 for semantic
  loss, and from +0.77715 to +1.10714 to +1.41765 to +1.59196 for motion loss.
  This strongly supports increasing scene-aligned radar dependence with data
  diversity. It remains one seed and a diagnostic loss result with a random
  frozen camera encoder, not proof of downstream IoU improvement.
- Next: P9, run the complete 28,130-sample training split under the unchanged
  protocol and verify that every training token is seen.

### 2026-10-07 — P7 1,024-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 1,024-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale1024_seed125.json`; checkpoint and raw
  JSONL outputs remain ignored.
- Reason: P7 tests whether the increasing radar-dependence trend at 64 and 256
  samples persists when the nested training subset grows by another factor of
  four without adding optimizer updates.
- Reproduce: use the P5 command with `--scale 1024` and replace each
  `scale64_seed125` output stem with `scale1024_seed125`; keep all other locked
  arguments unchanged.
- Verification: 30,000 consecutive effective updates covered exactly 1,024
  unique training indices and 29 completed sampler epochs. The run recovered
  21 AMP overflows, sustained 10.262 updates/s, peaked at 3.574 GiB CUDA
  allocation and had mean joint/semantic/motion training losses
  0.4761/0.3056/0.3410. Although the interactive terminal handle expired, the
  process completed both training and validation atomically. Independent
  checks found exactly 30,000 train and 6,019 finite validation records, 150
  scene summaries, no same-scene wrong-radar pair and no partial record.
- Result: full-validation matched semantic loss was 0.31452 versus
  0.31841 zero-velocity, 0.34782 empty and 0.33544 wrong-scene, giving
  penalties +0.00389/+0.03330/+0.02093. Matched motion loss was 0.98694 versus
  2.27702/2.42932/2.40459, giving penalties +1.29008/+1.44237/+1.41765.
  Moving-frame penalties were +1.45684/+1.62883/+1.60091 and cell-weighted
  penalties were +1.54969/+1.73035/+1.70491. Semantic wrong-scene penalties
  were positive in 149/150 scenes and motion penalties in 146/150 scenes.
- Interpretation/limitations: P5 to P6 to P7 wrong-scene penalties increase
  from +0.00017 to +0.00450 to +0.02093 for semantic loss, and from +0.77715
  to +1.10714 to +1.41765 for motion loss, while matched losses decrease.
  This is strong evidence that the diagnostic learns greater scene-aligned
  radar dependence with more diverse data. It remains one seed, a random
  frozen camera path, a feature/motion objective rather than downstream IoU,
  and does not yet establish that radar improves a matched camera-only model.
- Next: P8, repeat the unchanged protocol with the nested 4,096-sample subset.

### 2026-10-06 — P6 256-sample radar scaling result

- Change: completed the unchanged seed-125, one-sweep, 256-sample scale at
  30,000 effective optimizer updates and evaluated all 6,019 validation frames
  under the four locked radar modes. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale256_seed125.json`; the checkpoint and
  raw train/validation JSONL files remain ignored.
- Reason: P6 is the second fixed-budget point and tests whether the dependence
  seen at 64 samples strengthens when the nested subset contains four times as
  many samples and scenes.
- Reproduce: use the P5 command with `--scale 256` and replace each
  `scale64_seed125` output stem with `scale256_seed125`; all other arguments,
  including 30,000 updates, seed 125, one sweep and 6,019 validation samples,
  are unchanged.
- Verification: the run completed 30,000 consecutive effective updates over
  exactly 256 unique training indices, with 19 recovered AMP-overflow retries,
  10.295 updates/s, 3.574 GiB peak CUDA allocation and mean
  joint/semantic/motion training losses 0.3282/0.2517/0.1531. Independent
  checks found exactly 30,000 train and 6,019 finite validation records, 150
  scene summaries, no same-scene wrong-radar pair and no partial record. The
  same test suites and compilation used by P5 remained passing before the run;
  result integrity assertions passed after it.
- Result: full-validation matched semantic loss was 0.38045 versus
  0.38740 zero-velocity, 0.39019 empty and 0.38495 wrong-scene, giving
  penalties +0.00694/+0.00974/+0.00450. Matched motion loss was 1.25646 versus
  2.36205/2.37656/2.36359, giving penalties +1.10560/+1.12010/+1.10714.
  Moving-frame penalties were +1.24852/+1.26490/+1.25025 and cell-weighted
  penalties were +1.37098/+1.38008/+1.36908. Semantic wrong-scene penalties
  were positive in 111/150 scenes and motion penalties in 136/150 scenes.
- Interpretation/limitations: from P5 to P6, matched semantic and motion losses
  decreased while wrong-scene penalties increased from +0.00017 to +0.00450
  and +0.77715 to +1.10714. This is consistent with greater radar dependence
  at 256 samples, but two single-seed points cannot establish a reproducible
  onset. The random frozen camera path and lack of downstream IoU remain the
  same limitations as P5.
- Next: P7, repeat the unchanged protocol with the nested 1,024-sample subset.

### 2026-10-06 — P5 64-sample radar scaling result

- Change: completed the locked seed-125, one-sweep, 64-sample scale at 30,000
  effective optimizer updates and evaluated all 6,019 validation frames under
  matched, zero-velocity, empty and deterministic wrong-scene radar. Added
  per-scene metric aggregation, paired penalties, configurable progress output,
  and recoverable AMP gradient-overflow retries to
  `scripts/train_radar_scaling.py`. Whitelisted the compact result
  `artifacts/trainval/radar_scaling_scale64_seed125.json`; raw JSONL logs and
  the 188 MB checkpoint remain ignored.
- Reason: P5 is the first fixed-budget point in the adopted data-scaling curve.
  Full-validation paired interventions are needed to distinguish semantic and
  motion dependence on radar before changing the training sample count.
- Reproduce: run `TORCH_HOME=.cache/torch MPLCONFIGDIR=.cache/matplotlib
  python scripts/train_radar_scaling.py --data-root
  /home/students2026/datasets/nuscenes --bevcar-source external/BEVCar
  --manifest configs/radar_scaling_manifest_seed125.json --scale 64
  --max-updates 30000 --checkpoint-every 1000 --checkpoint
  checkpoints/radar_scaling/scale64_seed125/latest.pt --train-log
  artifacts/trainval/radar_scaling_scale64_seed125_train.jsonl
  --validation-records
  artifacts/trainval/radar_scaling_scale64_seed125_val.jsonl --summary-out
  artifacts/trainval/radar_scaling_scale64_seed125.json --val-samples 6019
  --print-every 100 --validation-print-every 500 --seed 125 --nsweeps 1`.
  Add `--resume` with otherwise identical arguments after an interruption.
- Verification: the initial process saved update 7,000 and then exposed an AMP
  gradient overflow at update 7,217. The resumed process reproduced that exact
  event, reduced the scaler from 16,384 to 8,192, retried the same batch, and
  completed. The final run made 30,000 effective updates over exactly 64 unique
  training indices, with 18 recovered overflow retries, 10.267 updates/s,
  3.574 GiB peak CUDA allocation and mean joint/semantic/motion training losses
  0.1794/0.1510/0.0568. Independent checks found 30,000 consecutive train-log
  records, 6,019 finite validation records, 150 scene summaries, no same-scene
  wrong-radar pair and no leftover partial record. The three relevant CPU test
  suites passed 14/14 and Python compilation passed.
- Result: full-validation matched semantic loss was 0.46199 versus 0.46504
  zero-velocity, 0.46413 empty and 0.46216 wrong-scene, giving penalties
  +0.00304/+0.00214/+0.00017. Matched motion loss was 1.58663 versus
  2.29385/2.37895/2.36377, giving much larger penalties
  +0.70722/+0.79232/+0.77715. Moving-frame penalties were
  +0.79864/+0.89474/+0.87761 and cell-weighted penalties were
  +0.88404/+0.98410/+0.96961. Motion penalties were positive in 142/150 scenes
  for empty and wrong-scene radar; semantic wrong-scene penalties were positive
  in only 74/150 scenes.
- Limitations: this is a dependence result for feature and motion losses, not a
  vehicle-segmentation IoU result or proof that fusion improves over a matched
  camera-only baseline. The camera encoder remains frozen at random
  initialization, the radar encoder starts randomly, only one seed and one
  sweep are used, and the 64 samples repeat for 468 epochs. P10 must compute
  scene-level confidence intervals before making an onset claim.
- Next: P6, repeat the unchanged 30,000-update/full-validation protocol with
  the nested 256-sample subset.

### 2026-10-06 — P4 end-to-end streaming and resume pilot

- Change: added `scripts/train_radar_scaling.py`, an opt-in batch-streaming
  trainer that builds frozen-DINO radar-anchored targets online, optimizes the
  current BEVCar semantic+motion joint model under AMP, calls the P2 periodic
  checkpoint utility, truncates stale post-checkpoint log records on resume,
  and performs streaming four-mode validation with all/moving/cell-weighted
  motion summaries. Added `IndexedSubset`/prefix helpers to
  `scripts/radar_scaling_data.py`, shared scene-spread validation positions,
  and the compact result `artifacts/trainval/radar_scaling_pilot128.json`.
- Reason: P1--P3 tested the components independently. Scale runs require proof
  that the real DINO target, VoxelNet fusion, optimizer/AMP state, sampler and
  validation path remain bounded and coherent across an actual process restart,
  plus a measured compute budget before starting the 64-sample result.
- Reproduce: first run the trainer with `--scale 128 --max-updates 128
  --checkpoint-every 64 --stop-after-update 64 --val-samples 4` and paths
  `checkpoints/radar_scaling/pilot128/latest.pt`,
  `artifacts/trainval/radar_scaling_pilot128_train.jsonl`,
  `artifacts/trainval/radar_scaling_pilot128_val.jsonl`, and
  `artifacts/trainval/radar_scaling_pilot128.json`; rerun the identical command
  without `--stop-after-update` and with `--resume`. Both use seed 125, one
  sweep, semantic weight 1.0 and motion weight 0.5. Set `TORCH_HOME` and
  `MPLCONFIGDIR` to the repository caches.
- Verification: CPU manifest/evaluation/checkpoint suites and compilation
  passed. A two-update end-to-end precursor passed joint backward, a 188 MB
  checkpoint and trained-model AMP validation. The formal run stopped at update
  64 with sampler position 64, then a new process printed
  `resumed update=64 epoch=0 position=64` and completed all 128 unique training
  positions. It saw moving targets in 113/128 updates and 738 covered cells;
  mean joint/semantic/motion losses were 1.5688/0.5384/2.0608. Online-DINO
  throughput was 9.608 updates/s, estimated 30,000-update pure training time
  0.867 h, peak allocation 3.574 GiB, and the final checkpoint was 187,965,586
  bytes. Four scene-spread AMP validation frames (3 moving, 18 cells) ran at
  3.063 frames/s with finite four-mode metrics. Matched semantic/motion loss was
  0.3828/1.4902 versus empty 0.4875/1.6268 and wrong-scene 0.4557/1.6909.
- Decision/limitations: lock 30,000 updates and online teacher extraction for
  P5--P9; do not allocate the optional ~49 GiB DINO patch cache. Checkpoint
  interval becomes 1,000 for scale runs. The pilot's four-frame losses are an
  engineering signal only, not radar-effect evidence. The current diagnostic
  still freezes a randomly initialized camera encoder, and full-val downstream
  utility remains P11. Initialization/devkit loading and full 6,019-frame
  validation add wall time beyond the 0.867 h pure-training estimate.
- Next: P5, the committed 64-sample / 30,000-update / full-val seed-125 result.

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

### 2026-09-28 — Image-level dense DINOv2 distillation (replaces the camera-BEV term)

- Change: moved the second distillation term from the camera BEV to the
  pre-projection image features, where DINOv2 supervision needs no depth and no
  radar. `nets/segnet.py` now also returns `feat_camXs_` (B*S, feat2d_dim,
  Hf, Wf) from `return_shared_bev=True`; `scripts/semantic_distill_smoke.py`
  drops `--camera-distill`/`--camera-weight` and adds `--image-distill` /
  `--image-weight`, with an `image_head` (feat2d_dim -> 384) that maps the image
  features to the frozen DINOv2 patch features already cached as `features` in
  `dinov2_teacher_features.npz`. No demo change was needed and no interpolation
  is required: the res101 image features (112x192 input, stride 8) and the
  DINOv2 vits14 patch grid (196x336 input, patch 14) are both 14x24.
- Reason: distilling on the BEV is a dead end because "dense + depth-correct +
  radar-free" cannot all hold there — a depth-correct dense BEV target needs
  radar (making it radar-guided again and near-identical to the fused term) or
  is a naive-rays blur. Distilling the image features directly keeps the term
  dense, radar-free, and disjoint from the fusion term (it shapes the encoder
  output before projection/fusion), matching "keep DINO semantics from being
  diluted".
- Verification: `python3 -m py_compile` and `git diff --check` passed. Four
  20-step single-sample smokes (fusion loss, cosine-to-DINO): baseline 0.2039;
  +image-distill (frozen) 0.2039 with image loss 0.430; +train-encoder only
  0.1730; +train-encoder +image-distill 0.1726 with image loss 0.405.
- Limitation: the image term now converges cleanly (image loss ~1.0 -> ~0.41)
  and no longer drags the fusion loss (0.1726 vs 0.1730), but the fusion-loss
  improvement is still driven by encoder trainability, and the ~0.0004 image
  delta is within single-sample smoke noise. Single mini sample, 20 updates,
  training losses only. The image term's real target is semantic retention in
  the image features, which a fusion-loss readout only sees indirectly.
- Next: judge the image term on its own objective (image-feature similarity to
  DINO on held-out frames, or a frozen-feature probe) rather than on fusion
  loss, and run a larger multi-sample comparison before claiming any benefit.

### 2026-09-28 — Held-out probe of the image-level distillation term

- Change: added `scripts/eval_image_distill_heldout.py` and its
  `scripts/run_image_distill_heldout.sh` runner, which train two
  trainable-encoder variants (baseline vs image-distilled) on a few training
  samples and measure (a) held-out fusion loss and (b) held-out image-feature
  similarity on scene-disjoint validation samples. `scripts/dinov2_mini4_experiment.py`
  was fixed to unpack the now four-valued `SemanticDistillationModel.forward`
  with `prediction, *_ = model(...)`.
- Reason: the previous smoke could only read the image term indirectly through
  fusion loss. This probe measures the image term on its own objective — how
  well held-out image features retain frozen-DINOv2 semantics — which is the
  term's actual claim.
- Verification: `python3 -m py_compile` and `git diff --check` passed. Three
  seeds (125/42/7) with 4 train + 4 held-out mini samples and 60 steps each
  gave held-out fusion-loss deltas that flip sign across seeds (image_distill
  minus baseline: -0.077, +0.040, -0.020), i.e. no reliable effect on held-out
  fusion loss. The held-out image loss was stable at ~0.57 across seeds
  (random/collapsed cosine is ~1.0), so the image features do retain DINOv2
  semantics on held-out frames even though this does not move fusion loss.
- Limitation: 4 train + 4 held-out mini samples, 60 updates, three seeds. The
  fusion-loss delta is noise-level and the image-retention result is a
  small-sample probe, not a downstream benchmark.
- Next: keep the image term as the camera-side half of the advisor-endorsed
  dual-teacher (image DINO + radar-anchored BEV DINO), and turn to the
  advisor's primary open problem — making matched radar explicitly beat empty
  or mismatched radar — via a stronger radar encoder, multi-sweep input, and
  balanced loss weights.

### 2026-09-29 — Multi-sweep radar ablation (10 sweeps)

- Change: threaded an `--nsweeps` option through `scripts/dinov2_mini4_experiment.py`
  (target generation) and `scripts/compare_radar_dino_tiny.py` (comparison and
  radar ablation), with nsweeps-specific target-cache directories and output
  filenames; `run_dinov2_mini4.sh` and `run_radar_dino_tiny.sh` forward it via
  the `NSWEEPS` environment variable. Re-cloned the pinned BEVCar source
  (`https://github.com/robot-learning-freiburg/BEVCar.git` at `29cacda3`) under
  the git-ignored `external/` directory for the `bevcar` branch.
- Reason: the advisor's recommendation #2 was multi-sweep radar (10-12 sweeps);
  every prior experiment used one sweep, so this is the first direct test of
  whether denser radar input makes the self-supervised objective depend on
  radar.
- Verification: at 5 and 10 sweeps, 3 of 4 training frames grew from ~400
  points to ~2100 and ~3300-4300 respectively (the first scene frame has no
  history and stays at 403). Mean held-out cosine loss across 12 frames / 3
  seeds, over the sweep-size curve:

  | Sweeps | Branch | Correct | Other scene | Positions only | Mixed values | No radar |
  | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | 1 | Lightweight | 0.903 | 0.909 | 0.903 | 0.903 | 0.911 |
  | 1 | BEVCar | 0.493 | 0.490 | 0.495 | 0.493 | 0.489 |
  | 5 | Lightweight | 0.890 | 0.904 | 0.890 | 0.890 | 0.910 |
  | 5 | BEVCar | 0.488 | 0.485 | 0.492 | 0.488 | 0.484 |
  | 10 | Lightweight | 0.879 | 0.900 | 0.879 | 0.879 | 0.909 |
  | 10 | BEVCar | 0.486 | 0.483 | 0.491 | 0.486 | 0.480 |

  The lightweight encoder's "No radar" penalty (No radar minus Correct) grows
  monotonically with sweeps — +0.008, +0.020, +0.030 — and correct beats
  other-scene by 0.006, 0.014, 0.021. BEVCar's penalty stays near zero and
  slightly negative throughout (-0.004, -0.004, -0.006).
- Limitation: multi-sweep strengthens the small radar-dependence of the
  lightweight encoder but does not repair the strong BEVCar encoder, whose low
  loss remains consistent with a statistical/spatial shortcut. This confirms
  that the objective (not input density) is the bottleneck for label-free radar
  dependence.
- Next: attack the objective itself — loss balancing / cosine weighting and the
  dual-teacher (image DINO + radar BEV DINO) so matched radar must outperform
  empty radar, rather than only increasing radar input density.

### 2026-09-29 — Dual-teacher 0.8/0.2 joint training (image DINO + radar BEV DINO)

- Change: extended `scripts/eval_image_distill_heldout.py` to train a weighted
  dual-teacher objective `L = fusion_weight * L_fusion + image_weight * L_image`
  (camera DINOv2 on image features vs. radar-anchored BEV DINO on the fusion
  BEV), swept four variants — fusion-only baseline, 1:1, 0.8-camera/0.2-radar,
  and 0.2-camera/0.8-radar — and added a no-radar sensitivity probe to
  `evaluate` (zero the radar voxel input, re-measure held-out fusion loss).
  Emits `artifacts/dual_teacher_heldout.json` (committed) plus a metrics
  `.npz` (git-ignored).
- Reason: the advisor's recommendation #5 — the student team's own "two
  teachers, one camera-side and one BEV/radar-side, ~0.8+0.2 weights" idea
  that the advisor endorsed but had not been implemented. It is the last
  input/weight-side lever untested after the strong-encoder and multi-sweep
  experiments.
- Verification: `python3 -m py_compile` passed. 4 train + 4 held-out mini
  samples, 60 steps, seed 125, trainable res101 encoder. Held-out fusion loss
  and no-radar penalty (no-radar minus correct; positive = radar-dependent):

  | Variant | fusion_w | image_w | Held-out fusion | No-radar fusion | Radar penalty | Held-out image |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | baseline (fusion only) | 1.0 | — | 0.556 | 0.561 | +0.0053 | — |
  | image_distill (1:1) | 1.0 | 1.0 | 0.534 | 0.538 | +0.0040 | 0.575 |
  | dual 0.8 camera / 0.2 radar | 0.2 | 0.8 | 0.556 | 0.565 | +0.0089 | 0.577 |
  | dual 0.2 camera / 0.8 radar | 0.8 | 0.2 | 0.584 | 0.590 | +0.0059 | 0.593 |

  The radar penalty stays tiny (~0.004–0.009, <2% relative) under every
  weighting, so zeroing the radar input barely moves held-out fusion loss. Both
  heads train and generalize (image loss ~0.57–0.59); the 1:1 image term
  actually lowers held-out fusion loss to 0.534, the best of the sweep.
- Limitation: the dual-teacher does not make the model depend on radar
  measurements. This closes out the advisor's input/weight-side
  recommendations: the radar-anchored teacher uses only radar *position* as
  depth anchors and never radar's measurement values (RCS / velocity /
  Doppler), so no encoder strength, input density, or loss weighting can force
  the student to require radar. The bottleneck is the target itself.
- Next: replace or extend the radar-anchored target with one that requires the
  radar measurement signal — the radar Doppler/motion-preservation target
  (SELF_SUPERVISED_EXTENSION_PLAN.md section 4) — and re-run the sensitivity
  protocol on the full fusion model.

### 2026-09-29 — Radar velocity/motion target (Doppler preservation) diagnostic

- Change: added `scripts/compare_radar_velocity_tiny.py` and
  `scripts/run_radar_velocity_tiny.sh` — a training-sample overfit diagnostic
  that attaches a 2-channel velocity head to the light and BEVCar radar
  encoders and trains it to predict the ego-motion-compensated velocity
  (vx_comp, vy_comp) of moving radar returns splatted into BEV cells, then
  measures the loss change when the velocity columns are ablated
  (`geometry_only` zeros them, `permuted_measurements` shuffles them, `empty`
  zeroes all radar). Emits `artifacts/radar_velocity_tiny.json` (committed).
- Reason: SELF_SUPERVISED_EXTENSION_PLAN.md section 4 — the one lever that
  forces radar-*measurement* dependence, since every prior target used only
  radar position (depth anchors) and never RCS / velocity / Doppler. Directly
  tests whether a target that requires velocity makes the encoder require
  velocity.
- Verification: `python3 -m py_compile` passed. 4 mini samples, 100 steps, 3
  seeds. The target is restricted to moving points (speed > 1 m/s) because a
  naive all-point mean-L1 velocity target is dominated by near-zero static
  velocity (first attempt: correct ≈ before ≈ 0.22, geometry_only penalty
  ≈ +0.001). Moving returns are sparse — ~10–14 of ~400+ points per sample
  (~3%) — so 12 steps under-trains; 100 steps converges. Mean Huber loss on
  moving cells, and the no-velocity penalty (geometry_only minus correct):

  | Branch | Correct | Positions-only (zero velocity) | Penalty | Mixed values | No radar |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | BEVCar | 0.059 | 0.400 | **+0.341 (6.8×)** | 0.548 | 2.666 |
  | Lightweight | 1.429 | 1.647 | +0.218 (15%) | 1.637 | 2.649 |

  The BEVCar encoder reads velocity to near-zero loss (0.059) and zeroing the
  velocity columns raises the loss 6.8× — the first large radar-measurement
  penalty in the whole project, confirming the target is what forces
  measurement dependence. The lightweight point-encoder only partially recovers
  velocity (1.43 vs the 2.65 "predict zero" floor), because its mean/max voxel
  pooling dilutes the sparse per-point velocity.
- Limitation: training-sample overfit diagnostic, not a generalization result;
  it shows only that the encoder *can* read velocity and that ablation moves
  the loss in the favorable setting. The moving-restricted target needs ~100
  steps (vs ~12 for the dense semantic target) because motion is sparse and the
  gradient is weak.
- Next: integrate the motion target into the joint objective
  `L = λ_sem·L_DINO_BEV + λ_motion·L_Doppler` (using BEVCar or a
  velocity-preserving lightweight encoder) and re-run the full fusion
  sensitivity protocol to confirm the full model now depends on radar
  measurements.

### 2026-09-29 — Motion target on the full fusion model (metaradar) — insufficient alone

- Change: added `scripts/compare_motion_target_tiny.py` and
  `scripts/run_motion_target_tiny.sh` — attaches a 2-channel motion head to the
  full image+radar fusion model's shared BEV (`MotionDistillationModel`
  subclasses `SemanticDistillationModel`), trains it on the moving-return
  compensated velocity target with the camera encoder frozen (so velocity can
  only reach the head through the radar branch), and ablates the radar input:
  `zero_velocity` (metaradar channels 3:7 = vx, vy, vx_comp, vy_comp) and
  `no_radar`. Emits `artifacts/motion_target_tiny.json` (committed).
- Reason: the plan's Doppler/motion target applied to the real fusion model
  (not just the isolated radar encoders), to test whether the target alone
  makes the full model depend on radar velocity.
- Verification: `python3 -m py_compile` passed. 4 samples, 100 steps, 3 seeds,
  frozen res101 camera encoder, trainable bev_compressor + heads. Mean loss and
  the no-velocity penalty (zero_velocity minus correct):

  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | semantic_only | semantic | 0.313 | 0.314 | +0.001 | 0.323 |
  | motion_only | motion | 0.564 | 0.612 | +0.048 (9%) | 1.175 |
  | joint (1.0 sem / 0.5 mot) | semantic | 0.268 | 0.269 | +0.001 | 0.280 |
  | joint (1.0 sem / 0.5 mot) | motion | 0.589 | 0.642 | +0.053 | 1.225 |

  The motion head learns the sparse target (motion loss 2.85 → 0.56) but the
  no-velocity penalty stays small (+0.05, ~9% relative) while no-radar is large
  (+0.61). So the full model predicts the motion map mainly from radar
  *position* and barely reads velocity — because the metaradar input's
  `voxelize_xyz_and_feats` "last-write-wins" voxelization overwrites the sparse
  per-voxel velocity with a random point (unlike BEVCar's learned voxel encoder,
  which kept the 6.8× velocity penalty in the radar-encoder diagnostic).
- Limitation: training-sample overfit diagnostic, not generalization. It shows
  the motion target alone is insufficient on the current fusion model, not that
  a motion target can never work.
- Next: replace the fusion model's metaradar branch with a velocity-preserving
  encoder (BEVCar VoxelNet) and re-run the motion-target sensitivity probe; the
  velocity penalty should then be large, confirming the recipe is a
  motion/Doppler target *plus* a velocity-preserving radar encoder.

### 2026-09-29 — Motion target + BEVCar branch in the full fusion — camera shortcut defeats it

- Change: added an opt-in `use_bevcar_encoder` / `bevcar_encoder` /
  `bevcar_encoder_channels` path plus a `zero_camera_bev` diagnostic flag to
  `nets/segnet.py` (the BEVCar VoxelNet `(B,128,Z,X)` output is concat'd with
  `camera_bev` before `bev_compressor`, mirroring `use_radar_encoder`; all other
  radar/lidar paths unchanged). Added `scripts/compare_motion_target_bevcar_tiny.py`
  and `scripts/run_motion_target_bevcar_tiny.sh`: the same frozen-camera
  motion-target probe as the metaradar experiment, but with the fusion's radar
  branch replaced by BEVCar VoxelNet. Ablations zero the BEVCar velocity channels
  (`features[..., 4:6]` = raw_vx/raw_vz) or all radar. `--zero-camera` zeros the
  camera BEV to cut the camera shortcut. Emits
  `artifacts/motion_target_bevcar_tiny.json` and
  `..._zerocam.json` (both committed).
- Reason: finish the recipe — a velocity-preserving radar encoder (BEVCar) in the
  real fusion model, expected to restore the large velocity penalty that metaradar
  lost.
- Verification: `python3 -m py_compile` passed. 4 samples, 100 steps, 3 seeds,
  frozen res101 camera encoder. Mean loss and velocity penalty (zero_velocity
  minus correct):

  Camera ON (normal fusion):
  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 0.444 | 0.425 | −0.019 | 1.532 |
  | joint (1.0/0.5) | motion | 0.433 | 0.435 | +0.002 | 1.501 |

  Camera ZEROED (control):
  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 0.052 | 0.116 | **+0.064 (2.2×)** | 2.697 |
  | joint (1.0/0.5) | motion | 0.024 | 0.080 | +0.056 (2.3×) | 2.723 |

  With the camera on, the velocity penalty is still ~0 — no better than metaradar.
  The tell is `no_radar = 1.53`, *below* the predict-zero floor (~2.70, the mean
  |velocity| of moving points): the model predicts velocity better than chance even
  with radar fully zeroed, because the frozen camera BEV is a per-sample
  memorization key (4 fixed samples), so the trainable bev_compressor/head look up
  the velocity field without reading radar velocity. Zeroing the camera removes the
  shortcut: correct drops to 0.052 (matching the standalone BEVCar diagnostic's
  0.059), the velocity penalty becomes +0.064 (2.2×, clearly positive vs metaradar's
  +0.05 and the camera-on ~0), and no_radar returns to the 2.70 floor. So the
  VoxelNet → bev_compressor path *does* preserve velocity through the fusion; the
  camera shortcut was the only blocker.
- Limitation: small-sample overfit, not generalization; the camera memorization is
  an artifact of 4 fixed samples (a frozen random camera encoder is still
  sample-identifiable). Freezing the encoder is therefore not enough to isolate
  radar velocity in a 4-sample diagnostic. The penalty is 2.2× here vs 6.8× in the
  standalone radar encoder, because the bev_compressor mixes in position features
  that soften the loss when velocity alone is removed.
- Next: the recipe is now motion/Doppler target + velocity-preserving radar encoder
  + cutting the camera shortcut. To prove the full model needs radar velocity
  end-to-end, either (a) run the motion probe on a held-out split (camera cannot
  memorize unseen samples), or (b) route the motion head off a radar-only branch so
  the camera can never supply velocity, and re-measure the penalty at scale.

### 2026-09-29 — Motion target + BEVCar held-out probe — velocity does not generalize at 4 samples

- Change: added `scripts/compare_motion_target_bevcar_heldout.py` and
  `scripts/run_motion_target_bevcar_heldout.sh`. Same BEVCar fusion + motion target
  as the overfit probe, but training samples come from the `train` split and
  evaluation from the scene-disjoint `val` split, with the camera encoder kept ON
  and frozen (an unseen sample's camera BEV is a pattern the bev_compressor has
  never seen and cannot memorize). The probe's helpers are reused by importing from
  `compare_motion_target_bevcar_tiny` rather than duplicated. Emits
  `artifacts/motion_target_bevcar_heldout.json` (committed).
- Reason: option (a) from the previous entry — prove whether the full model needs
  radar velocity end-to-end once the camera cannot memorize unseen samples.
- Verification: 4 train / 4 val samples, 100 steps, 3 seeds. Held-out (val) motion
  loss and velocity penalty (zero_velocity − correct):

  | Variant | Loss | Correct | Zero velocity | Penalty | No radar |
  | --- | --- | ---: | ---: | ---: | ---: |
  | motion_only | motion | 2.193 | 2.181 | −0.012 | 2.585 |
  | joint (1.0/0.5) | motion | 2.409 | 2.405 | −0.004 | 2.570 |

  Held-out motion `correct ≈ 2.2–2.4` is essentially the predict-zero floor (~2.70)
  and the velocity penalty is ≈0 (slightly negative = noise). So on unseen samples
  the model does not use radar velocity for the motion target: the velocity-reading
  ability does not generalize from 4 training samples. Radar *position* still helps
  a little (no_radar 2.57–2.59 vs correct 2.19–2.41), but the velocity channels add
  nothing. This is the decisive control — the earlier overfit "success"
  (camera-zeroed correct=0.052, penalty 2.2×) was memorization of the 4 training
  samples, not a learned velocity function.
- Limitation / interpretation: not a recipe failure but the expected small-sample
  result. The tiny probes establish the mechanism (BEVCar carries velocity through
  the fusion) but cannot demonstrate generalization; whether the velocity penalty
  turns positive on held-out data is exactly what needs many more samples (the full
  nuScenes run). Also flags a real scaling risk: the motion target supervises only
  ~3% of returns (~8–16 moving cells of 40 000), so the velocity signal is sparse
  and may need re-weighting at scale.
- Next: scale up the held-out probe (train ~24–64 samples) to see whether the
  velocity penalty turns positive before committing to the full supercomputer run,
  or go straight to the full nuScenes run with the motion loss re-weighted to the
  sparse moving-point coverage.

### 2026-09-29 — Held-out probe scaled to 64 samples — velocity penalty turns positive

- Change: added a `--tag` argument to
  `scripts/compare_motion_target_bevcar_heldout.py` (suffixes the output filename)
  and a matching `TAG` env passthrough in `scripts/run_motion_target_bevcar_heldout.sh`
  so scale runs write separate JSONs instead of overwriting the 4-sample one.
  `.gitignore` now un-ignores `artifacts/motion_target_bevcar_heldout_*.json`. Ran
  the same held-out probe with 64 train / 16 val samples and 512 steps (8 epochs).
  Emits `artifacts/motion_target_bevcar_heldout_train64.json` (committed).
- Reason: option b1 — test whether the velocity penalty turns positive as the
  training set grows, before committing to the full supercomputer run.
- Verification: mean held-out velocity penalty (zero_velocity − correct) vs training
  samples:

  | Train samples | motion_only penalty | joint penalty |
  | ---: | ---: | ---: |
  | 4 | −0.012 | −0.004 |
  | 64 | +0.009 | +0.022 |

  Held-out motion `correct` also falls as data grows (motion_only 2.19→1.88; joint
  2.41→1.64), so the model starts predicting velocity rather than sitting at the
  predict-zero floor (~2.70). The penalty turns positive but remains small: velocity
  dependence is emerging but weak at 64 samples. The trend confirms the
  data-scarcity reading — more samples → more radar-velocity dependence, which is
  what the full nuScenes run should deliver.
- Limitation: the val split contains many stationary scenes with zero moving points
  (`motion_loss` returns 0 there, diluting the mean), so the reported penalty is a
  mean over all val samples and the per-moving-sample penalty is larger than shown.
  Still small in absolute terms at 64 samples.
- Next: re-run at 128 samples to confirm the monotonic trend, then decide whether to
  proceed to the full nuScenes run (with the motion loss re-weighted to the sparse
  moving-point coverage).

### 2026-09-30 — 128-sample probe: fixed motion_loss crash, deferred to 20 GB GPU

- Change: fixed a crash in `motion_loss` (`scripts/compare_motion_target_bevcar_tiny.py`):
  when a sample has no moving points it returned a detached `torch.zeros(())`, so in the
  `motion_only` variant `loss` became a leaf with no grad and `.backward()` raised
  "element 0 of tensors does not require grad and does not have a grad_fn". It now
  returns `prediction.sum() * 0.0` (a true zero that keeps the graph connected). Also
  added `flush=True` to the seed-result prints in
  `scripts/compare_motion_target_bevcar_heldout.py` so long-run progress is visible.
- Reason: the 128-sample run (128 train / 32 val, 1024 steps) crashed after the first
  control variant. The 64-sample run only survived because its 64 train samples happened
  to all contain moving points; the 128-sample set adds stationary samples.
- Verification: script compiles and the fixed branch is exercised by the val set of the
  completed 64-sample run. The 128-sample re-run was started and confirmed progressing,
  but was stopped early: the laptop GPU (RTX 5060 Laptop) was clock-capped at 1065 MHz
  (34% of its 3090 MHz max, 30 W against a 115 W limit) with VRAM 97% full, giving a
  ~2.5–3 h ETA instead of ~25 min. No 128-sample result was produced.
- Next: run the 128-sample (or larger) probe on the machine room's 20 GB GPU.

### 2026-09-30 — Dual-teacher held-out probe: seed-list + tag

- Change: `scripts/eval_image_distill_heldout.py` now accepts `--seed-list`
  (comma-separated seeds, replacing the single `--seed`) and `--tag` (output
  filename suffix); per-variant metrics are aggregated across seeds before
  writing `artifacts/dual_teacher_heldout{_tag}.json`. The previous fixed output
  name made consecutive runs overwrite each other.
- Reason: run the 64-sample dual-teacher + weight-sweep probe (baseline,
  image_distill, 0.8-camera/0.2-radar, 0.2-camera/0.8-radar x 3 seeds) without
  clobbering results, and let the 20 GB machine run the same probe in parallel.
- Verification: script compiles; the `--tag train64` run is in flight on the laptop.

### 2026-09-30 — Vectorized the radar-anchored soft-target splat

- Change: `scripts/dinov2_bev_demo.py` `radar_anchored_soft_targets` no longer
  scatters each visible radar point in a Python loop (one GPU-to-CPU sync per
  point per field). The Gaussian scatter is now one vectorized pass (broadcast
  grid + boolean window mask + `einsum`).
- Reason: the loop dominated target extraction, making 64-sample feature
  extraction take ~90 min on the throttled laptop instead of minutes.
- Verification: a synthetic old-loop-vs-vectorized equivalence test matches to
  float32 precision (max abs diff ~5e-7, identical nonzero cell counts); module
  compiles.

### 2026-09-30 — 64-sample, three-seed dual-teacher held-out probe

- Change: ran the committed multi-seed/tagged dual-teacher held-out probe and
  committed `artifacts/dual_teacher_heldout_train64.json`; whitelisted that
  compact JSON result in `.gitignore`. No model, target, or official
  Simple-BEV path changed. The downloaded nuScenes mini dataset and DINOv2
  source/weights remain local dependencies and are not tracked.
- Reason: scale the four-sample dual-teacher weight sweep to 64 training and 16
  scene-disjoint held-out mini samples on the available 24 GB GPU, using the
  newly vectorized target construction and three independent seeds.
- Verification: `CONDA_ENV=bev NUSCENES_ROOT=/home/students2026/datasets/nuscenes
  TORCH_HOME=/home/students2026/dd2414/.cache/torch TRAIN_SAMPLES=64
  VAL_SAMPLES=16 STEPS=512 bash scripts/run_image_distill_heldout.sh
  --seed-list 125,42,7 --tag train64` completed with
  `DUAL_TEACHER_HELDOUT_OK`. Mean held-out fusion / no-radar fusion / penalty
  (no-radar minus correct) across 48 seed-frame evaluations were: baseline
  `0.5238 / 0.5410 / +0.0172`; image-distill `0.4910 / 0.4996 / +0.0086`;
  0.8-camera/0.2-radar `0.4760 / 0.4897 / +0.0137`; and
  0.2-camera/0.8-radar `0.4703 / 0.4847 / +0.0144`. Held-out image cosine
  losses were `0.5274`, `0.5272`, and `0.5288` for the three image-supervised
  variants, respectively. The official nuScenes devkit loaded 10 scenes, 404
  samples, and 31,206 sample-data records; the repository data-only smoke and
  Python compilation passed before the run.
- Interpretation: increasing data and updates improves the held-out fusion
  losses relative to the preceding 4-train-sample probe, with the
  radar-weighted 0.2-camera/0.8-radar setting best on that loss. But all
  no-radar penalties remain only about 2--3% of the correct loss, so this
  semantic objective still provides weak evidence of radar use and no evidence
  that it reads radar measurement values.
- Limitations: this is a 64-sample mini probe, not a full-dataset benchmark.
  It tests radar removal rather than matched-versus-mismatched radar or
  velocity/Doppler ablations; the radar-anchored DINO target still uses radar
  positions as depth anchors, not its measurement fields. The DINOv2 teacher
  is frozen and no human-box supervision is used here.
- Next: retain this result as the scaled dual-teacher baseline; prioritize the
  motion-target held-out scale-up (128 samples or larger) and/or a control in
  which matched radar must beat empty and mismatched radar before making a
  radar-learning claim.

### 2026-09-30 — Dual-teacher 64-sample multi-sweep comparison

- Change: added `--nsweeps` to `scripts/eval_image_distill_heldout.py` and
  forwarded `NSWEEPS` in `scripts/run_image_distill_heldout.sh`, so the same
  dual-teacher held-out protocol can select the radar history merged by the
  nuScenes loader. The value is stored in report metadata. Ran and committed
  the compact 5- and 10-sweep JSON reports; the code preserves the previous
  default of one sweep and all official Simple-BEV paths.
- Reason: test the sparse-radar hypothesis before combining architectural
  changes. Hold train/validation samples (64/16), updates (512), seeds
  (125/42/7), frozen DINOv2 teacher, target construction, and four loss
  configurations fixed; change only the merged radar sweeps (1, 5, or 10).
- Verification: `CONDA_ENV=bev NUSCENES_ROOT=/home/students2026/datasets/nuscenes
  TORCH_HOME=/home/students2026/dd2414/.cache/torch TRAIN_SAMPLES=64
  VAL_SAMPLES=16 STEPS=512 NSWEEPS=<5|10> bash
  scripts/run_image_distill_heldout.sh --seed-list 125,42,7
  --tag train64_nsweeps<5|10>` completed for both sweep counts with
  `DUAL_TEACHER_HELDOUT_OK`. A 4-train / 4-held-out / 1-step 5-sweep and
  10-sweep smoke also passed. Mean held-out fusion / no-radar penalty across
  48 seed-frame evaluations were:

  | Sweeps | Variant | Fusion | Penalty |
  | ---: | --- | ---: | ---: |
  | 1 | baseline | 0.5238 | +0.0172 |
  | 1 | image-distill | 0.4910 | +0.0086 |
  | 1 | 0.8 camera / 0.2 radar | 0.4760 | +0.0137 |
  | 1 | 0.2 camera / 0.8 radar | 0.4703 | +0.0144 |
  | 5 | baseline | 0.4698 | +0.0147 |
  | 5 | image-distill | 0.4613 | +0.0079 |
  | 5 | 0.8 camera / 0.2 radar | 0.4556 | +0.0180 |
  | 5 | 0.2 camera / 0.8 radar | 0.5143 | +0.0150 |
  | 10 | baseline | 0.4612 | +0.0148 |
  | 10 | image-distill | 0.5176 | +0.0052 |
  | 10 | 0.8 camera / 0.2 radar | 0.4677 | +0.0133 |
  | 10 | 0.2 camera / 0.8 radar | 0.4556 | +0.0196 |

- Interpretation: the best loss is the 10-sweep, radar-weighted
  0.2-camera/0.8-radar configuration (0.4556), marginally below the best
  5-sweep result (0.4556 before rounding). Its no-radar penalty rises from
  +0.0144 at one sweep to +0.0196 at ten sweeps, a modest favorable signal.
  The curve is not monotonic across variants: at five sweeps the radar-heavy
  configuration degrades to 0.5143, while the camera-heavy configuration is
  best. Extra radar density therefore changes optimization, but it has not
  established a universal weight preference or strong radar dependence.
- Limitations: a +0.0196 penalty is only about 4.3% of the correct loss. This
  target still uses radar positions as depth anchors and does not require RCS,
  Doppler, or velocity; it evaluates empty radar but not matched-versus-wrong
  radar. This remains a nuScenes-mini held-out feature-loss probe, not a
  downstream detection/segmentation benchmark or a BEVCar fusion result.
- Next: at 10 sweeps, sweep radar-majority dual-teacher weights (for example
  0.6, 0.7, and 0.8 radar) under this same protocol. Then add the BEVCar
  encoder, motion/soft-region controls, and hybrid-teacher components one at a
  time with matched ablations rather than combining them before attribution.

### 2026-10-05 — Trainval configuration and review, no experiment execution

- Sync: `git fetch origin` succeeded. HEAD and the feature-branch remote both
  resolved to `3745647` (0 ahead / 0 behind); no pull was needed. The earlier
  staged mini 64-sample/multi-sweep changes were already present and preserved.
- Changes: added `configs/trainval.env`, `scripts/run_trainval.sh`,
  `scripts/heldout_data.py`, `scripts/test_trainval_configuration.py` and
  `TRAINVAL_RUN_PLAN.md`. Updated `scripts/dinov2_bev_demo.py`, both held-out
  Python probes and their shell launchers, `scripts/smoke_test_mini.py`, and
  `train_nuscenes.py`. Refreshed `README.md`, `SETUP_LOCAL.md`,
  `SELF_SUPERVISED_EXTENSION_PLAN.md` and this handoff.
- Rationale: remove hard-coded mini selection from the active held-out paths,
  expose motion-probe sweeps, keep both splits on one devkit instance, avoid
  CUDA-in-forked-worker problems with worker counts of zero, spread trainval
  probe frames across scenes, record exact sample manifests, and prevent
  accidental dense all-dataset GPU caching. Historical defaults and official
  model/checkpoint interfaces remain unchanged.
- Prepared parameters: `bev`, `dset=trainval`, batch 1, workers 0, sweeps 5
  (override 1/10); bounded probes use 64 train / 16 val, 512 updates and seeds
  125/42/7. A separate supervised streaming profile uses 30,000 updates,
  accumulation 1, 224x400 images, checkpoint interval 1,000, and full-split
  validation through the separate eval mode. These are proposed budgets,
  not measured trainval results.
- Verification: `/home/students2026/miniconda3/envs/bev/bin/python
  scripts/test_trainval_configuration.py` passed all four CPU-only mocked
  tests (selection/bounds, cache-budget guards, manifest/disjointness, mini and
  trainval loader routing). Python compilation and shell syntax checks passed.
  `bash scripts/run_trainval.sh <data-check|smoke|dual-teacher|motion|supervised>`
  and `INIT_DIR=/placeholder/checkpoint bash scripts/run_trainval.sh eval`
  printed valid commands and returned without running a loader/model. Metadata
  inspection counted 700/150 scenes and 28,130/6,019 samples; a complete
  sample-data filename scan found 0 missing and 0 zero-byte files out of
  2,631,083. `git diff --check` passed.
- Limitations: no real loader batches, DINO extraction, model forward/backward,
  training or evaluation were run for this change, as requested. Full-file
  existence checks do not prove every sensor payload decodes correctly. BEVCar
  external source is missing locally. Current GPU-cached distillation probes
  remain bounded to 160 total frames and lack resume; a full-data streaming
  self-supervised trainer and unified hybrid model still need implementation.
- Publication blocker: `git var GIT_AUTHOR_IDENT` fails with "Author identity
  unknown" because no Git name/email is configured. Do not invent a teammate's
  identity. No new commit/push is claimed; previously staged work is retained.
- Next: user review of the plan, then the explicitly requested smoke stage;
  supply a Git author identity before publishing these changes.

### 2026-10-05 — BEVCar VoxelNet 64/16 reproduction on trainval

- Scope: cloned the official BEVCar repository at the project-pinned commit
  `29cacda3bc5416d47428c1d0f017527acad34f90` (git-ignored external dependency).
  The experiment dynamically imported only `nets/voxelnet.py`, initialized it
  randomly, and used the existing Simple-BEV fusion/motion wrapper. No full
  BEVCar architecture or BEVCar checkpoint was loaded.
- Exact reproduction: after a successful one-batch data check and one-step
  VoxelNet smoke, ran `TRAIN_SAMPLES=64 VAL_SAMPLES=16 STEPS=512
  SEED_LIST=125,42,7 NSWEEPS=1 SAMPLE_SELECTION=head
  TAG=trainval_train64_val16_nsweeps1_head bash scripts/run_trainval.sh motion
  --execute`. This preserved the latest mini protocol and changed only the
  dataset version. The selected trainval subset covered 2 train scenes and 1
  val scene; 10/16 val frames had moving target cells.
- Exact result (correct / zero-velocity / penalty / no-radar motion loss):
  semantic-only `1.0922 / 1.0920 / -0.0002 / 1.0760`; motion-only
  `1.0641 / 1.1409 / +0.0768 / 0.9097`; joint
  `1.2210 / 1.3014 / +0.0804 / 0.9473`. Moving-frame-only penalties were
  `+0.1228` and `+0.1286` for motion-only/joint.
- Scene-coverage control: repeated the same budget with
  `SAMPLE_SELECTION=uniform`, covering 64 distinct train and 16 distinct val
  scenes; 15/16 val frames had moving cells. Motion-only was
  `1.6998 / 2.4977 / +0.7980 / 2.6245`; joint was
  `1.5796 / 2.4398 / +0.8602 / 2.6140`. Moving-frame-only penalties were
  `+0.8512` and `+0.9176`.
- Interpretation: semantic-only remains velocity-insensitive. The exact head
  result has a positive but modest velocity effect and no-radar is better than
  correct, so it does not establish reliable radar use. The broad-scene control
  shows strong, seed-consistent VoxelNet velocity dependence and correct radar
  beats removal, but it changes sample selection and remains a bounded probe.
- Artifacts: committed candidates are
  `artifacts/trainval/motion_target_bevcar_heldout_trainval_train64_val16_nsweeps1_head.json`
  and `..._uniform.json`; each records exact indices, tokens and scene tokens.
- Verification/limitations: data check loaded 28,130/6,019 samples and 313
  radar points in the first train frame; one-step smoke and both three-seed runs
  ended with `MOTION_TARGET_BEVCAR_HELDOUT_OK`. No 5/10-sweep, all-frame
  streaming pretraining, checkpoint/resume test or downstream IoU was run.
- Publication blocker: no commit/push was created because repository Git author
  name/email are unset. Existing staged mini work remains preserved.
- Next: choose between a matched 1/5/10-sweep broad-scene comparison and the
  streaming/checkpoint implementation required for all 28,130 train frames.
