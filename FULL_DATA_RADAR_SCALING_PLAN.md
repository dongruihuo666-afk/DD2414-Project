# Full-data radar scaling study

## Decision and research question

This plan records the adopted path for testing whether radar becomes useful as
the amount and diversity of nuScenes training data increase. The primary study
uses the official `v1.0-trainval` split, BEVCar's pinned `nets/voxelnet.py`, one
radar sweep, and seed 125. It measures a deterministic nested data curve:

```text
64 -> 256 -> 1,024 -> 4,096 -> 28,130 training samples
```

Every scale is evaluated on the same complete 6,019-frame validation split.
The curve distinguishes three claims that must not be conflated:

1. radar values reach the network (`zero_velocity` versus matched radar);
2. the network uses scene-aligned radar (matched versus empty and wrong-scene
   radar); and
3. radar improves an independent downstream task (camera-plus-radar versus a
   matched camera-only model on vehicle-segmentation IoU).

The first curve addresses claims 1 and 2. A downstream camera-only comparison
is the acceptance gate for claim 3. Self-supervised losses alone are not a
downstream performance result.

## Locked study protocol

The protocol is frozen before the first scale result. Any later change requires
a documented protocol revision and a restart of affected comparisons.

| Item | Locked choice |
| --- | --- |
| Dataset | Official nuScenes `v1.0-trainval`; train 28,130 / val 6,019 keyframes; official scene-disjoint split |
| Radar encoder | Only `external/BEVCar/nets/voxelnet.py` at commit `29cacda3bc5416d47428c1d0f017527acad34f90`; random initialization; no BEVCar checkpoint |
| Radar history | One sweep for the primary curve; 5/10 sweeps are deferred until the one-sweep result is interpretable |
| Sensors | All five nuScenes radar sensors, using the existing project geometry and velocity rotation |
| Seed | 125 for the full curve; seeds 42 and 7 only for confirmation of the onset point and full-data result |
| Data subsets | Deterministic, nested and scene-spread; never the first contiguous frames; manifests record indices, sample tokens and scene tokens |
| Training budget | Fixed budget of 30,000 optimizer updates per scale, batch size 1. P4 measured 9.61 online-DINO joint updates/s (about 0.87 h pure training per scale), so the provisional budget is now locked. This lets the full run see every training frame while holding compute constant. |
| Objective | Freeze one objective and its weights before the curve. Initial candidate: semantic weight 1.0 plus motion weight 0.5. Do not tune weights separately per scale. |
| Validation | All 6,019 validation frames, deterministic order, no parameter updates |
| Radar evaluation modes | `matched`, `zero_velocity`, `empty`, and deterministic `wrong_scene` |
| Primary diagnostics | Semantic loss, motion loss, moving-frame-only motion loss, cell-weighted motion loss, and paired deltas from matched radar |
| Final utility metric | Full-validation vehicle-segmentation IoU from matched camera-only and camera-plus-radar downstream protocols |

The current held-out probe freezes a randomly initialized camera encoder. That
configuration is acceptable only for a bounded radar-path diagnostic. Before
claiming that the complete fused representation is useful, the camera encoder
must receive a meaningful, matched training signal (for example frozen-DINOv2
image-level distillation), and the camera-only and fused variants must share the
same initialization and update budget.

## Required implementation boundaries

- Keep the official Simple-BEV training and checkpoint interfaces unchanged.
- Add the full-data path as an opt-in experiment rather than increasing the
  existing GPU-cached probe limit.
- Stream one batch at a time. Never materialize all dense `(384, 200, 200)` BEV
  targets; that cache would require roughly 977 GiB before inputs or optimizer
  state.
- Cache compact FP16 DINOv2 patch features only if the throughput benchmark
  justifies it. A complete train+val cache is expected to be roughly 49 GiB and
  must be keyed by sample token, camera order, resolution and teacher identity.
- Checkpoints, teacher caches, logs and raw per-frame predictions remain outside
  Git. Commit only code, configuration, manifests and compact JSON summaries.
- A completed milestone updates `PROJECT_HANDOFF.md`, records exact commands and
  observed results, passes proportionate checks, and is committed and pushed on
  `dd2414-mini-baseline` before the next milestone begins.

## Milestone and commit table

Status values are `not started`, `in progress`, `blocked`, and `complete`.

| ID | Status | Goal | Required artifact and acceptance test | Commit boundary |
| --- | --- | --- | --- | --- |
| P0 | complete | Freeze and publish this plan | This file, README link, and handoff update agree on scope, controls, scale order and publication workflow; Markdown and `git diff --check` pass | `docs: plan full-data radar scaling study` |
| P1 | complete | Deterministic nested manifests and streaming batches | Manifests for 64/256/1,024/4,096/full are nested, scene-spread, reproducible and train/val-disjoint; iterate at least 512 batches without retaining prior GPU tensors | `feat: add streaming radar scaling data path` |
| P2 | complete | Periodic checkpoint and exact resume | Save student, optimizer, GradScaler, epoch/update, sampler position and Python/NumPy/Torch/CUDA RNG; interrupted-resumed smoke matches an uninterrupted run within the documented numerical tolerance | `feat: add resumable self-supervised checkpoints` |
| P3 | complete | Matched/zero/empty/wrong-scene validation | Wrong radar always comes from a different scene under a deterministic token map; all four modes cover identical validation tokens and produce compact per-frame records | `feat: add wrong-scene radar evaluation` |
| P4 | complete | End-to-end throughput and stability pilot | 128-sample train / fixed dev pilot runs, resumes once, shows bounded memory, measures samples/s and cache/disk cost, and locks or revises the provisional 30,000-update budget before P5 | `test: validate full-data radar training pipeline` |
| P5 | not started | Run the 64-sample scale | Seed 125 training completes under the locked protocol; evaluate all 6,019 val frames in four radar modes; record exact manifest and paired metrics | `exp: record 64-sample radar scaling result` |
| P6 | not started | Run the 256-sample scale | Same acceptance criteria and unchanged protocol as P5 | `exp: record 256-sample radar scaling result` |
| P7 | not started | Run the 1,024-sample scale | Same acceptance criteria and unchanged protocol as P5 | `exp: record 1024-sample radar scaling result` |
| P8 | not started | Run the 4,096-sample scale | Same acceptance criteria and unchanged protocol as P5 | `exp: record 4096-sample radar scaling result` |
| P9 | not started | Run the complete 28,130-sample scale | Every train token is seen under the locked sampler; resume is exercised in the real run; all 6,019 val frames receive four-mode evaluation | `exp: record full-data radar scaling result` |
| P10 | not started | Analyze the dependence curve | One compact report plots data scale against velocity/empty/wrong-scene penalties, includes per-scene uncertainty and identifies whether an onset is repeatable rather than a single noisy point | `docs: analyze radar dependence scaling curve` |
| P11 | not started | Confirm radar utility with downstream IoU | Matched camera-only and fused checkpoints use identical labeled probe/fine-tune budgets; report full-val IoU under matched/empty/wrong radar; repeat the onset and full scale with seeds 42 and 7 if the seed-125 result is positive | `exp: validate radar utility on full-val IoU` |

## Metrics and interpretation

For each validation sample and loss `L`, store paired values and summarize:

```text
velocity penalty = L(zero_velocity) - L(matched)
empty penalty    = L(empty)         - L(matched)
wrong penalty    = L(wrong_scene)   - L(matched)
```

Positive velocity penalty shows that the model reads radar velocity. Positive
empty penalty shows reliance on some radar information. Positive wrong-scene
penalty shows use of the correct camera-radar association. These are dependence
tests, not proof of task improvement.

Aggregate motion both over all frames and over moving frames/cells. A frame with
no moving target cells must be reported as zero coverage rather than silently
diluting the moving-frame statistic. Report per-scene paired means and a
scene-level bootstrap 95% confidence interval; do not treat every temporally
adjacent frame as an independent observation.

The downstream claim requires all of the following:

1. fused matched-radar IoU exceeds the matched camera-only IoU;
2. matched radar exceeds empty radar in the same fused checkpoint;
3. matched radar exceeds wrong-scene radar in the same fused checkpoint; and
4. the direction is consistent across the confirmation seeds or its per-scene
   confidence interval excludes zero.

If only motion loss reacts to zero velocity, conclude that radar is encoded but
has not been shown to improve vehicle segmentation. If empty is worse but wrong
radar matches correct radar, conclude that the model uses radar statistics but
not alignment.

## Per-milestone publication checklist

Before each milestone commit:

1. inspect `git status` and preserve unrelated files;
2. update the status row in this plan;
3. update `PROJECT_HANDOFF.md` with changed paths, rationale, exact reproduction
   command, actual result, limitations and next milestone;
4. run the milestone's tests plus compilation/shell syntax and
   `git diff --check` as applicable;
5. stage only source, config, manifests and compact summaries;
6. commit on `dd2414-mini-baseline`, push it, and verify that the remote branch
   resolves to the local commit; and
7. do not begin the next milestone while the completed one is uncommitted or
   unpushed, unless the handoff explicitly records the external blocker.

Large checkpoints, DINO caches, logs and raw predictions are recoverable local
artifacts and must not be committed.
