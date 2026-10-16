# Full-data multi-sweep, double-distillation follow-up

Date: 2026-10-16. Branch: `dd2414-mini-baseline` only.
This supersedes the pending-task order of `MEETING_FOLLOWUP_DAY_PLAN.md`.
The cluster experiment branch and the official Simple-BEV path are unchanged.

## What was missing and what is being added

| Meeting request | Previous queue | New action |
| --- | --- | --- |
| Official pretrained segmentation reference | Both complete, 6,019 validation frames | Keep 448x800 references; additionally evaluate the same checkpoints at 112x192 |
| Trainable camera DINO + radar-masked fused-BEV DINO | Not in the local legacy run | Independent full-data double-distillation pretraining at five and ten sweeps |
| Depth supervision and BEV geometry | Depth illustrations only | Train a metric depth head with radar anchors; actually weight image-to-voxel lifting by predicted depth |
| Depth Anything V2 | Cached source/weights; illustrations only | Additional treatment: relative inverse-depth distillation plus metric radar anchors |
| Frozen versus unfrozen downstream | Historical one-sweep finetuning running | Same pretrained checkpoint: frozen 1x1, unfrozen 1x1, and fully unfrozen official decoder |
| Radar density | Legacy one/five sweeps, different objectives | New matched five/ten-sweep pair, fixed teacher target support and all other settings |
| Independent vehicle motion | Annotation coverage audit only | Train a linear object-velocity readout on frozen BEV features at GT box ROIs; evaluate four radar interventions |
| Reliable unattended execution | False legacy probe idle timeout | New serial service: no idle/elapsed/24-hour cutoff; durable resume, bounded failure retries, incremental results |

The low historical IoU has NOT been causally attributed to camera supervision
or radar sparsity. Those are hypotheses. The historical frozen probe, the
unfrozen official decoder and new double-distillation probes differ in several
ways and must not be reported as a single-factor experiment.

## Model and losses

Six cameras -> trainable random-initialized ResNet-101 dense encoder -> camera
features -> original Simple-BEV geometric voxel sampling -> BEV compressor.
Five physical radar sensors, each aggregating **five or ten chronological
sweeps**, feed the separately pinned BEVCar VoxelNet. Camera/radar BEV features
are concatenated before the shared BEV compressor. Ten sweeps does not mean
ten physical radar sensors.

Both new treatments use the same 7,000-occupied-voxel safety bound, matching
the loader's largest possible ten-sweep point tensor (700*10). The previous
3,500 guard could reject a dense ten-sweep frame. VoxelNet already accepts
dynamic occupied counts, so this changes no network parameters, grid size or
retained spatial evidence; no points are dropped to fit a smaller guard.

A frozen DINOv2 ViT-S/14 teacher supervises both:

1. Dense camera image features, before geometric projection.
2. Fused BEV features, only in radar-anchored soft regions.

The primary pretraining loss is `0.2 * image DINO + 0.8 * masked BEV DINO`.
The existing radar-cell motion reconstruction remains an auxiliary term with
weight 0.5. It is not an independent vehicle-motion score. No segmentation or
box-velocity labels are used by the pretraining objectives.

To isolate input density in the new five/ten-sweep pair, **both** use returns
with absolute relative timestamp <= 0.1 seconds for teacher BEV targets and
motion auxiliary targets. All accumulated returns still enter VoxelNet. The
same recent-anchor policy is used for metric radar-depth supervision. This
also limits moving-object ghosts from stale returns. It does not establish
that aggregating old returns is motion-compensated or always beneficial.

The opt-in depth head predicts bounded log camera-Z, 0.5–120 meters. Its
SmoothL1 loss at projected recent radar returns has weight 0.2. A Gaussian
likelihood in log depth (sigma 0.6) weights the sampled camera features along
each ray before camera BEV aggregation. GT depth is not fed into inference.
The original `Segnet` implementation and checkpoint format are unchanged;
only this experiment's voxel utility is wrapped.

Depth Anything V2 Small is frozen. Its standard weights predict **relative**
depth, not metric depth. The added weight-0.05 shape loss compares normalized
inverse student metric depth against the relative teacher output, invariant
to positive affine changes of the teacher's output. Radar supplies metric
anchors. Teacher inference uses 196x336 images, not the official default
518-pixel input; quality at this lower teacher resolution is a limitation.

The depth treatment changes both supervision and lifting. A later loss-only
versus lifting-only ablation is needed to separate these mechanisms.

## Fixed first-pass protocol and queue order

Use the seed-125 manifest, all **28,130 train / 6,019 validation** frames,
scene-disjoint splits, six 112x192 cameras, the 200x8x200 voxel grid, BF16,
AdamW, 0.25-epoch warmup and cosine decay. The shared batch size is locked only
after all CUDA engineering gates pass; batch five failed a depth-aware CUDA
engineering check, so batch four is the common candidate. Do not silently
shrink a failing scientific run. All ten CUDA cases subsequently passed at
batch four; the 100-update heaviest-case throughput pilot measured 6.632
samples/s and 18.276 GiB peak allocation, then a fresh process resumed at
update 100 and advanced to 101 / sampler cursor 404. **Batch four is locked**,
with eight workers. Every new experiment uses five or ten
sweeps. Single-sweep results are historical controls, not a new strict
one/five/ten-sweep comparison under these new objectives.

1. Safely finish the already-started historical raw-velocity finetuning from
   its saved checkpoint, retaining the original three-epoch configuration.
2. Recover the five-sweep historical frozen probe that the previous watchdog
   incorrectly terminated. That legacy probe has no mid-epoch resume.
3. Evaluate official RGB and RGB+radar checkpoints at 112x192; preserve the
   previously completed 448x800 reference scores separately.
4. Double DINO + five sweeps, no depth: pretrain, frozen 1x1, full finetuning,
   unfrozen 1x1, independent object-motion readout.
5. Repeat exactly with ten sweeps.
6. Repeat at five sweeps with radar-depth supervision and depth-aware lifting.
7. Repeat at five sweeps with radar depth plus Depth Anything V2 shape loss.

The initial screening budget is **one complete pretraining epoch** and **one
complete epoch for each independent downstream readout/finetuning run**.
Every downstream job loads the same completed pretraining checkpoint for its
configuration, never the preceding downstream job's weights. Both 1x1 probes
start with identical head initialization and use official `SimpleLoss(2.13)`.
The frozen backbone stays in eval mode; its full parameters/buffers hash must
remain unchanged. The unfrozen probe changes trainable features but keeps the
same 1x1 head; full finetuning uses the larger official segmentation/center/
offset decoder and official multitask loss. Camera LR is 1e-5 during downstream
unfreezing, other student LR 1e-4 (2e-4 for the unfrozen-1x1 run), and readout
LR 1e-3. Pretraining LR is 2e-4.

This is an equal-epoch full-data **screen**, not a convergence/capability
ceiling. A short linear probe may underfit. Do not compare one new pretraining
epoch against eight legacy epochs as a clean image-SSL ablation. Longer
epoch-matched training and multiple seeds are the next evidence boundary.
The new pipeline does not launch a new single-sweep velocity-zeroed job;
that pending old-queue task is deferred under the multi-sweep priority.

Preliminary total queue estimate is about 20–28 hours, not a guaranteed
deadline. This extrapolates the local pilot and prior downstream timings;
ten-sweep I/O, full validation and object-level controls can change it. A
healthy task continues beyond a day rather than being forcibly truncated.

## Evaluation

- Primary segmentation metric: full-split valid-cell intersection / union
  with `sigmoid().round()` exactly as in Simple-BEV. Report Vehicle IoU,
  precision, recall, F1, TP/FP/FN/TN and predicted-positive fraction. This is
  binary vehicle segmentation, not multi-class mIoU.
- Keep GT/probability/prediction/error PNGs for the same preselected tokens,
  and per-frame JSONL on the complete validation split.
- Official 448x800 scores are 47.247% RGB and 55.495% RGB+radar. They are
  supervised references, not proof of an architectural gain. Evaluating their
  checkpoints at 112x192 removes the inference-resolution difference but not
  their original high-resolution supervised training advantage.
- Vehicle motion: pool nine oriented interior points per GT box from frozen
  shared-BEV features; fit a linear 128->2 readout on the training split.
  GT box locations are supplied, but target velocity never enters the head.
  Targets are finite `nuScenes.box_velocity` temporal annotation estimates,
  rotated global->ego->reference-camera XZ axes, in m/s.
- Evaluate matched, velocity-zeroed, empty and wrong-scene radar with identical
  images, GT ROIs, readout and backbone. The wrong-scene input is a deterministic
  full-val cross-scene bijection. Report velocity endpoint error, vector RMSE
  and speed MAE, including moving (>1 m/s) and stationary groups. Compare
  zero-prediction and training-mean baselines. These are oracle-location
  readout metrics, **not** detection mAVE/NDS or a prediction of future motion.
  These interventions test sensitivity of the same trained fusion model;
  empty-radar inference is not an independently trained camera-only baseline.
  Zeroing velocity channels also leaves geometric multi-sweep motion trails.
- Radar-depth MAE/RMSE/AbsRel are errors against independent held-out radar
  measurements, not dense ground-truth depth. No claims about unseen test data.

## Run, monitor and resume

Only one GPU worker is active at a time. Do not start this launcher while the
old primary or recovery service is active. The operator's migration sends a
checkpoint request to the original queue first; no process is killed because
it is slow. The new queue resumes the historical finetuning with unchanged
hashed source files and optimizer/sampler/RNG state.

```sh
bash scripts/start_multisweep_followup.sh
systemctl --user status dd2414-multisweep-teacher.service
tail -F fullsize_baseline/runs/multisweep_teacher_20261016/queue.log
watch -n 10 /home/students2026/miniconda3/envs/bev/bin/python \
  scripts/monitor_baseline_followup.py \
  --study-dir fullsize_baseline/runs/multisweep_teacher_20261016
```

Progress/results: `pipeline.json`, `REPORT.md`, `results.json`; each stage has
`console.log`, `status.json`, `metrics.jsonl`, `tensorboard/`, `checkpoints/`,
`validation/epoch001/` and `summary.json`. Main runs are under the study path
above; recovered historical runs retain their original directory. Checkpoints
are atomic, every 250 updates and at epoch/stop boundaries, and preserve model,
optimizer, scaler, acknowledged sampler cursor and RNG states. Resume restores
prefetch-safe sample order; CUDA backward need not be bitwise deterministic.

For TensorBoard:

```sh
/home/students2026/miniconda3/envs/bev/bin/tensorboard \
  --logdir fullsize_baseline/runs/multisweep_teacher_20261016 --port 6006
```

There is **no** log-idle, stage-duration, 24-hour or systemd runtime kill rule
in the new service. A stale log produces a warning only. A genuine worker
failure resumes from its last durable checkpoint, with at most three attempts
per stage, then records failure and continues independent configurations.
Dependent jobs require completed pretraining. No automatic batch/epoch/sample
reduction. GitHub publication errors are recorded without stopping training.
Re-run the same launcher after an external failure; if its transient unit
remains registered, inspect it and reset its failed state before relaunching.
Do not use SIGKILL to stop training. An explicit service stop requests a safe
checkpoint; a full validation pass may finish before stopping.

Stage summaries/handoff are committed and pushed on the baseline branch with
explicit paths only. The unrelated staged project PDF, datasets, checkpoints,
raw features and logs are excluded. A power outage, disk failure or lost GPU
driver can still interrupt a job. The queue is independent of chat/API quota,
but automatic infrastructure repair is not guaranteed. A long queue may run
beyond a day; pending/partial work must remain labelled as such.

## Reproduction checks

Local external dependencies (not committed): BEVCar VoxelNet commit
`29cacda3bc5416d47428c1d0f017527acad34f90`, SHA256
`43ed0ee30f70c328bd47885cbc68272e7f5e3625c61b325b54f713a851f3e6ac`;
DINOv2 ViT-S/14 weights SHA256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
See `SETUP_LOCAL.md` for the existing environment/data/teacher cache.

Depth Anything V2 source is the official
`https://github.com/DepthAnything/Depth-Anything-V2` checkout at
`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` under `.cache/Depth-Anything-V2`.
The standard Small weights are
`https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth`,
stored as `.cache/depth_anything_v2_vits.pth`, SHA256
`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
The installed workstation environment already imports the required torch,
torchvision, OpenCV, NumPy and TensorBoard modules. No environment replacement,
new compiled extension or cluster deployment is performed by this task.

Actual engineering results: 57 discovered CPU unit tests pass; new Python
modules compile and launcher shell syntax passes. Ten real CUDA engineering
paths pass at batch four, including save/restart, four pretraining variants,
all downstream modes, depth-aware finetuning and depth-aware frozen probing.
Each completed engineering run uses only 8 train / 4 validation frames and
two updates, never evidence of model quality. The object-motion test evaluated
the same 55 annotated vehicles in four validation frames in every radar mode.
Largest observed allocated memory in those checks is 18.275 GiB. The failed
batch-five depth check is retained locally as failed, not relabelled passed.

```sh
PYTHONPATH=scripts /home/students2026/miniconda3/envs/bev/bin/python -m unittest \
  scripts.test_multisweep_teacher scripts.test_baseline_followup \
  scripts.test_fullsize_training scripts.test_fullsize_linear_probe \
  scripts.test_training_checkpoint
bash -n scripts/start_multisweep_followup.sh
/home/students2026/miniconda3/envs/bev/bin/python scripts/check_multisweep_teacher_cuda.py \
  --output-dir fullsize_baseline/smoke/multisweep_teacher_validation \
  --data-root /home/students2026/datasets/nuscenes \
  --bevcar-source external/BEVCar --batch-size 4
```

The CUDA checker uses 8 train / 4 val frames only to test engineering paths,
including a real process restart. Such results are not experiment evidence.
