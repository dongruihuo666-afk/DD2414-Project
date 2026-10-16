# Meeting follow-up: one-day baseline experiment plan

Date: 2026-10-16. Branch: `dd2414-mini-baseline`.
Status: execution approved. The opt-in implementation and CUDA gates are now
prepared; see the 2026-10-16 execution entry in `PROJECT_HANDOFF.md` and the
live queue status below for actual progress.

**Execution update:** the owner's later request supersedes the pending queue
order with `MULTISWEEP_TEACHER_DAY_PLAN.md`: local double distillation, matched
five/ten sweeps, actual depth-supervised lifting and GT-object motion readouts.
The already-started historical finetuning is checkpointed and will be resumed;
no new single-sweep jobs are added. The replacement queue has no idle/elapsed
termination rule. The older timed queue below is historical, not the launcher
to use for this expanded study.

## Meeting priorities and scope

First establish how the official pretrained Simple-BEV models perform in our
data/evaluation pipeline. Then test whether supervised downstream training
improves our existing representation when camera, radar, fusion, and decoder
are unfrozen. Depth supervision and independent vehicle-motion evaluation
follow these checks. The teacher explicitly described vehicle-motion prediction
as a validation proposal, not a required replacement pretraining loss.

Use the local baseline and existing checkpoints. Do not merge, cherry-pick,
edit, or submit `experiment/hybrid-teacher-10sweep-cluster`. Do not repeat the
completed eight-epoch one- or five-sweep pretraining.

## Verified starting point

Local summaries inspected on 2026-10-16:

| Existing experiment | State | Matched semantic loss | Matched radar-cell motion loss | Downstream evidence |
| --- | --- | ---: | ---: | --- |
| One sweep, velocity supplied, eight epochs | complete | 0.273902 | 0.088137 | Frozen one-epoch probe Vehicle IoU 4.962% |
| One sweep, velocity zeroed, eight epochs | complete | 0.279375 | 1.568412 | Frozen one-epoch probe Vehicle IoU 4.848% |
| Five sweeps, velocity supplied, eight epochs | complete | 0.269462 | 0.107164 | No five-sweep probe summary found |

These checkpoints have a randomly initialized, frozen camera encoder. They
already use DINOv2 BEV targets; camera image-feature DINO distillation is
absent. Low probe IoU has not been causally attributed to one choice. Radar-cell
motion loss may reward recovering velocity supplied in the input; it does not
independently establish useful vehicle-motion prediction.

One-sweep raw-velocity pretraining took 6.75 hours plus 2.17 hours of four-mode
validation. Five-sweep pretraining took 8.86 plus 2.80 hours. The raw-velocity
frozen probe took 1,422 seconds (23.7 minutes) for one full train epoch and full
validation. These rates do not predict unfrozen-camera backward cost.

At the initial planning inspection, official RGB/radar weights were absent
and the sandbox could not see CUDA. Both weights have since been downloaded
and strict-loaded on the actual RTX 3090 Ti. Full-val official references are
complete: 47.247% RGB and 55.495% RGB+radar at 448x800. This is verified host
execution, not a CPU-only CUDA claim.

## Evaluation contract

- Use all 28,130 train and 6,019 validation frames and the existing scene-disjoint
  seed-125 manifest. Bounded batches are engineering checks, not subset training.
- Primary Vehicle BEV IoU follows `eval_nuscenes.py`: sum valid-cell intersection
  and union over the full split, then divide. Preserve its
  `torch.sigmoid(logits).round()` threshold and tie behavior. Mean per-frame IoU
  is a separately named diagnostic, not this official aggregation.
- Report precision, recall, F1, TP/FP/FN/TN, predicted-positive fraction, epoch,
  and wall time. Keep threshold 0.5 fixed; any later calibration uses a train-only
  calibration split. Do not tune on final validation labels.
- Match ROI, GT/valid masks, camera reference, and visibility handling. Reconcile
  official/custom loader differences before claiming a fair numerical comparison.
- Official reference: six cameras, 448x800 images, 200x200 BEV, ResNet-101;
  radar reference uses the official metaradar path, five sweeps, and upstream
  filtering. Load the full model including its existing segmentation head.
  Simple-BEV already implements segmentation; no new detector conversion is needed.
- New legacy downstream experiments keep their actual historical 112x192 resolution
  and one sweep. Label this resolution difference. Optionally also evaluate
  official weights at 112x192; our own 448x800 run requires a separate budget.
- Render GT, probabilities, predictions, and TP/FP/FN for fixed validation scene
  tokens chosen independently of performance. Scene-bootstrap intervals estimate
  evaluation uncertainty, not training-seed variation.

The upstream README reports 47.6% camera-only and 55.8% camera+radar under its
documented settings. Keep these reference values separate from actual locally
reproduced scores; they are not guaranteed outcomes.

## Experiment matrix

| Experiment | Initialization | Trainable modules | Budget / purpose |
| --- | --- | --- | --- |
| Official pretrained camera | Official RGB checkpoint | None | Full validation; camera reference |
| Official pretrained camera+radar | Official radar checkpoint | None | Full validation, five sweeps; fusion reference |
| Existing one-sweep frozen probes | Raw/zero velocity epoch-8 checkpoints | Only existing 1x1 head | Reuse completed one-epoch results |
| Five-sweep frozen probe | Completed five-sweep epoch-8 checkpoint | New 1x1 head | One full train epoch + full validation; test density benefit |
| Full supervised finetuning, velocity supplied | Raw-velocity epoch-8 checkpoint | Camera, VoxelNet, fusion, original decoder | Three full epochs; full validation at epochs 1 and 3 |
| Full supervised finetuning, velocity zeroed | Zero-velocity epoch-8 checkpoint | Same modules | Identical schedule; zero input velocity throughout |
| Same-head freeze diagnostic, optional | Raw-velocity epoch-8 checkpoint | Identical 1x1 head; backbone either frozen or trainable | One full epoch; separate the freeze question from head capacity |

Full decoder finetuning uses the original Simple-BEV segmentation, center,
offset, and uncertainty losses. DINO and radar-motion heads do not participate
in downstream updates. The historical shared-BEV pretraining did not train
the original decoder: record its initialization, and use identical initialization
seeds/distributions across the paired runs. Do not introduce official supervised
BEV weights into these two legacy-initialized finetuning runs.

A full decoder and a frozen 1x1 head differ in capacity, losses, and trainable
parameters. Their gap measures downstream adaptability, not the isolated effect
of freezing. The optional same-head experiment isolates that factor. Raw/zero
checkpoints also have different pretraining histories; their difference compares
training recipes rather than a pure test-time velocity intervention.

## Relative 24-hour schedule

T0 starts when execution is approved and GPU access confirmed. Use one sequential
GPU queue. CPU tests and report preparation may overlap GPU work. Timings for
trainable-camera work are provisional until measured.

| Elapsed time | Work | Deliverable / gate |
| --- | --- | --- |
| 0-3 h | Audit weights/metrics; implement a separate baseline finetuning entrypoint and resumable runner; CPU tests and bounded forward/backward/resume checks | Strict load, finite gradients, exact split counts, frozen integrity, successful restored progress |
| 3-5 h | Full-val official RGB and radar evaluation at 448x800 | Two reference summaries, hashes, fixed-token masks |
| 5-6 h | Missing five-sweep frozen 1x1 probe with the historical one-epoch recipe | Full-val Vehicle IoU and one-versus-five table |
| 6-12 h | Three-epoch raw-velocity full decoder finetuning | Epoch-1/3 full-val metrics and periodic checkpoints |
| 12-18 h | Matched velocity-zeroed full decoder finetuning | Matched epoch-1/3 metrics and plots |
| 18-20 h | Optional same-1x1-head unfrozen diagnostic | Controlled head-capacity comparison; reuse frozen result only if all settings match |
| 20-22 h | Optional depth diagnostic and object-motion data audit | Depth/radar projection examples, velocity target coverage/units; defer if earlier jobs overrun |
| 22-24 h | Complete validation, aggregate results, masks, analysis, reproduction notes, and scoped commits | Honest complete/partial/failed status and meeting-ready evidence |

At the first 200 stable unfrozen updates, measure samples/s, data wait, peak
memory, update time, and validation cost. Estimate one epoch as 28,130 divided
by measured samples/s plus checkpoint/validation overhead. Do not reuse the
frozen-camera rate. Start engineering pilots at batch 1-2 and 4-8 workers;
lock a fitting common physical/effective batch for both production runs.

Provisional optimizer: AdamW, camera LR 1e-5, radar/fusion/decoder LR 1e-4,
weight decay 1e-5, warmup 0.25 epoch, then cosine decay over three epochs.
These are starting settings, not tuned optima. If the pilot changes them,
record the decision before either production run and apply it to both.

If two three-epoch runs do not fit the measured budget, choose a common one-epoch
schedule before either production run. If an unexpected overrun happens later,
save partial checkpoints and compare only identical completed epochs. Do not
shorten only the second run or call a partial pass a full epoch. Defer depth/motion
first, then the same-head diagnostic, and reserve time for final validation.

If official weights cannot be obtained, record that failure and proceed with
the existing-checkpoint probe/finetuning path. Diagnose official evaluation
failures before making fair-baseline claims; do not silently replace a missing
full-val reference with a tiny-subset result.

## Implementation gates before unattended execution

Baseline has the legacy model, completed checkpoints, official evaluator,
frozen probe, and reusable checkpoint utilities. The new opt-in full-data
finetuning program is `scripts/train_baseline_downstream.py`; its sequential
runner is `scripts/run_baseline_followup_day.py`. The old frozen probe uses
`no_grad` feature extraction and saves at
epoch boundaries; changing `requires_grad` alone does not implement finetuning
or mid-epoch resume.

Use opt-in scripts and preserve old defaults. Test strict legacy loading,
gradient flow through camera/radar/fusion, label geometry, and inactive unused
heads. In frozen mode, use eval mode and compare parameters AND buffers before
and after training. Record the unfrozen model's BatchNorm policy; normalization
updates are part of that treatment.

The runner must outlive the chat session: use a persistent local user service,
atomic stage status, per-run logs, periodic checkpoints, sampler/RNG restoration,
bounded retries, and a 24-hour deadline with signal-triggered checkpointing.
Reserve time for complete final validation. A failed/partial job cannot be
marked complete. Verify user-service/GPU access in the actual host session.
Chat quota must not control job progression.

Monitoring exposes epoch fraction, frames processed, LR/losses, samples/s,
last progress timestamp, GPU utilization/memory, and ETA. Record actionable
failure on non-finite loss/gradients, OOM, repeated failed resumes, or progress
stale beyond measured validation/checkpoint time. Do not silently change batch,
precision, objective, or model after a failure in a controlled run.

## Depth and object-motion next steps

For the optional depth diagnostic, freeze Depth Anything V2 Small and visualize
its output with projected radar points on fixed training-only scenes. Standard
Depth Anything V2 checkpoints predict relative depth, not meters. Resolve
depth/inverse-depth convention, scale/shift, visibility, and valid masks before
selecting a distillation loss. Historical multi-sweep returns are not synchronized
dense depth GT: ego-motion alignment does not correct independent object motion.

The next depth training comparison should hold semantic DINO supervision fixed
and vary only the depth term: relative-depth teacher and/or valid current-sweep
radar range anchors. A lower depth loss alone is insufficient; acceptance still
uses full-val Vehicle IoU and FP/FN maps. This requires a new camera training path
and matched control, not a promised full-data result during this day.

For object motion, use independent nuScenes annotation-derived vehicle velocities
as targets; GT boxes/masks locate where to pool fused BEV features. GT velocity
must never enter the readout input. Transform all velocities to the same frame,
exclude unavailable/non-finite targets, and record coverage. This is motion
evaluation with oracle object locations, not object detection.

Later train the same small frozen-feature readout for clearly defined camera-only,
matched-radar, zero-velocity, and empty-radar comparisons. Include zero-motion
and aggregate-radar-velocity baselines. Report vector error in m/s, moving/static
groups, and zero/low/high radar-return groups. Object-level targets provide a
more useful check than reconstructing radar velocity, but do not guarantee a gain.

After these baseline questions, plan matched DINO camera distillation, depth,
and 1/5/10-sweep studies. The separate cluster implementation remains outside
this day's scope.

## Outputs and evidence

Create a fresh study directory beneath ignored `fullsize_baseline/runs/` with
descriptive job names containing model, sweeps, velocity policy, head, and epoch
budget. Preserve old results. Every summary records code/checkpoint/manifest
hashes, split counts, camera initialization, freeze/normalization policy, head,
input/BEV resolution, sweeps, velocity policy, supervision, optimizer groups,
effective batch, completed epoch, wall time, and IoU aggregation.

Commit compact report tables, figures, and reproduction commands after each
completed milestone. Keep weights, raw caches, datasets, and logs untracked.
Minimum scientific outputs: official full-val references when accessible,
five-sweep probe, and a matched finetuning pair at a common completed epoch.
Label optional work and partial experiments separately.

Sources: supplied transcript (00:18, 02:37, 04:10, 09:46, 18:31, 21:58,
26:49, 35:17); [Simple-BEV official README](https://github.com/aharley/simple_bev#pre-trained-models);
[Depth Anything V2 official README](https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models);
`eval_nuscenes.py`, `scripts/train_fullsize_linear_probe.py`,
`scripts/compare_motion_target_bevcar_tiny.py`, and `FULLSIZE_MEETING_VISUALS.md`.

## Execution and monitoring

The user approved continuous execution on 2026-10-16. GPU access is available
on the actual host outside the tool sandbox. The measured historical input
size was corrected from the planning typo 224x400 to 112x192. Official RGB and
radar checkpoints were downloaded and strictly loaded in four-frame CUDA
engineering checks. Those small checks are not full-validation results.

Start/resume the persistent queue from the repository root:

```bash
bash scripts/start_baseline_followup_day.sh
```

Inspect it without interrupting training:

```bash
systemctl --user status dd2414-baseline-followup-day.service
watch -n 10 /home/students2026/miniconda3/envs/bev/bin/python \
  scripts/monitor_baseline_followup.py \
  --study-dir fullsize_baseline/runs/meeting_followup_20261016
tail -F fullsize_baseline/runs/meeting_followup_20261016/queue.log
```

Each child also has `console.log`, `status.json`, checkpoints, and a final
summary. The study's `comparison/REPORT.md`, `metrics.json`, plots, and
`validation/epoch*/mask_*.png` are produced automatically. Each completed
stage publishes compact results and a handoff update to the baseline branch;
publication failures are recorded and do not terminate training. The unrelated
staged PDF is excluded from explicit-path commits.

The queue first measures 200 full-data training updates at the selected batch,
then locks a common 1- or 3-epoch schedule. Each training update acknowledges
consumed samples only after successful optimization. Checkpoints include
optimizer, scaler, sampler, model/buffers, and RNG states; camera projection
backward on CUDA is not guaranteed bitwise deterministic. The real interrupted
smoke restored the correct next update and finished, but its trajectory was
not bitwise identical to an independent continuous run. Do not call that
exact numerical equivalence.

Optional depth/object-motion work currently produces training-only relative
depth illustrations and annotation velocity coverage. Radar/depth alignment
and a trained vehicle-motion readout remain future work.

The first five-sweep probe attempt was incorrectly stopped by the queue's
legacy timestamp handling after 30 minutes. The watchdog now accepts real
status/log file updates. The persistent recovery service
`dd2414-baseline-followup-recovery.service` waits for the original service to
finish, then re-enters the same queue with its original absolute deadline,
reusing completed stages and retrying the unfinished probe. This preserves
the currently running finetuning without a GPU interruption.
