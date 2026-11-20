# Full-data double-distillation and metric-depth results

Training results verified on 16 November 2026; direct checkpoint evaluation
completed on 16 November and verified/published on 20 November. These are official
nuScenes **validation** results, not hidden-test results. Historical trained-readout
numbers are final-epoch scores unless marked as best validation epoch.
Percentage-point differences are treatment minus control.

## Direct pretrained-checkpoint results: no probe or GT adaptation

This is the owner's requested primary reporting protocol. Load the completed
50-epoch pretraining checkpoint strictly and run its archived segmentation
decoder as-is: no fitted probe, optimizer, further training or threshold tuning.
GT is used only to compute metrics after prediction.

| Pretrained checkpoint | Vehicle IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| DINO only, no metric-depth supervision | **1.9571%** | 2.0128% | 41.4258% | 3.8391% |
| DINO + metric-depth supervision | **1.6342%** | 1.6527% | 59.3254% | 3.2159% |

Both evaluate exactly **6,019 unique validation samples / 150 scenes**, with
448x800 RGB, ten sweeps and velocity, FP32/TF32 disabled, batch four and two
loader workers. Metrics use the original Simple-BEV-style global valid-cell
confusion counts and fixed `sigmoid().round()` decision. These share metric
definitions, not the supervised training conditions of official Simple-BEV.
Historical trained-readout runs below use BF16 instead; that precision difference
is explicit.

The direct readout severely overpredicts foreground: predicted vehicle cells
occupy **32.2087% / 56.1744%** of valid cells, while GT vehicles occupy **1.5650%**.
This explains the very low precision despite nonzero recall. Depth lowers raw
Vehicle IoU by 0.3229 pp in this readout, but this is not an isolated estimate
of its effect on useful semantic representations.

**Critical limitation:** both archived segmentation decoders exactly match the
same original random initialization. The pretraining losses taught the encoder,
radar/fusion representation and distillation/depth heads, not a GT vehicle
segmentation decoder. Thus these are genuine untouched-checkpoint outputs, not
GT-trained segmentation capability. Their low scores must not be presented as
proof that every useful pretrained feature is absent. The **50.89% / 45.94%**
figures below are a separate protocol with five additional supervised epochs.
Historical frozen probes are retained only as auxiliary records.

The [reviewed machine-readable record](../results/pretrained_direct_evaluation_20261116.json)
includes exact fractional metrics, TP/FP/FN/TN, source/checkpoint hashes, decoder
and before/after model hashes, coverage and hashes of the local evidence files.
Verification reproduced all counts from 6,019 archived per-frame records per arm,
recomputed all four metrics, freshly hashed both input checkpoints, and confirmed
unchanged model parameters/BN buffers. No new inference or training was performed
while publishing this record. Both evaluation subprocesses exited successfully.

Runtime artifacts remain ignored under `.cache/pretrained_direct_eval_20261116`:
`pipeline.json`, per-arm `summary.json`, `validation/per_frame.jsonl` and logs.
Evaluation took 1008.57 / 1011.05 seconds, completed at 15:08 UTC on 16 November.
The exact inference-only reproduction command is in
[the experiment protocol](EXPERIMENT.md); use the corresponding checkpoint hash
in the reviewed record and a fresh output directory. No full evaluation rerun
is required to inspect these published results.

## Historical supervised-adaptation and probe findings

Metric depth supervision improved both frozen linear probes, but reduced the
five-epoch end-to-end segmentation result. The DINO-only control reached
**50.89% Vehicle IoU**; the depth-supervised treatment reached **45.94%**, a
**4.95 percentage-point decrease**. Frozen fused-BEV IoU instead improved from
**7.43% to 11.95%**. Better linear readability and better supervised adaptation
are different outcomes in this experiment.

Both 50-epoch pretrains and all six downstream runs completed successfully.
The final backup/verification/cleanup job completed on **15 November at 01:59
CET**. The account has no queued/running jobs at the report's live check.

## Controlled setup

| Setting | Shared configuration |
| --- | --- |
| Dataset | Official 700 training scenes / 150 validation scenes; 28,130 / 6,019 samples |
| Student initialization | Random ResNet101 and student, seed 125; no ImageNet or official Simple-BEV student weights |
| Initialization equality | Identical initial student SHA-256 in both arms: `077315f67731a31e7d1389926d83e2bfdc220d9c4a3a0cf89740a042ab958587` |
| Camera / radar | Six cameras, RGB 448x800; BEVCar VoxelNet, ten radar sweeps, velocity supplied |
| DINO teacher | Frozen DINOv2 ViT-S/14; image distillation and radar-supported fused-BEV distillation |
| Student geometry | Both arms use an 80-bin, 1-80 m depth probability head for camera-to-BEV lifting |
| Pretraining | 50 total epochs; 281,300 optimizer updates; 1,406,500 training sample visits per arm |
| Distributed training | Five GPUs, batch one per rank / global batch five; no padding or dropped training samples |
| Optimization | AdamW, initial LR 0.0002, one-epoch warmup, cosine floor 0.1; BF16; clipping at 5 |
| Frozen probes | Same pretrained student frozen; separately train a 128-to-1, 1x1 convolution for 10 epochs at camera BEV and fused BEV |
| Supervised adaptation | Separate branch from the same pretrained student; all student components and full segmentation decoder train for five epochs |
| Validation | All 6,019 samples after each epoch; global valid-cell Vehicle IoU, fixed sigmoid/round threshold |

The **control** uses `0.2 * image DINO + 0.8 * fused-BEV DINO`.
The **treatment** adds `0.05 * projected metric-depth cross entropy`, with a
frozen Depth Anything V2 Metric VKITTI Small teacher and a 2 m target width.
Its loss is applied in the calibrated camera BEV volume before radar fusion.
Thus this is a comparison of **adding metric-depth supervision**, not removing
the student depth/lifting structure from one arm.

No motion-reconstruction objective or sparse radar-depth objective is active.
Downstream segmentation/probes use ground-truth labels; the frozen teachers are
discarded. The generic historical `config.losses` description in downstream
files is not a list of losses active in supervised mode.

## Final downstream results

All values in this table are percentages. Each row covers the same 6,019
validation frames. Vehicle IoU is binary foreground IoU, not multiclass mIoU.

| Readout | Arm | Vehicle IoU | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen camera-BEV linear probe, epoch 10 | DINO only | 0.0076 | 0.90 | 0.0076 | 0.0151 |
| Frozen camera-BEV linear probe, epoch 10 | DINO + metric depth | 4.87 | 25.61 | 5.67 | 9.28 |
| Frozen fused-BEV linear probe, epoch 10 | DINO only | 7.43 | 28.11 | 9.18 | 13.84 |
| Frozen fused-BEV linear probe, epoch 10 | DINO + metric depth | 11.95 | 40.42 | 14.50 | 21.35 |
| Full supervised segmentation, epoch 5 | DINO only | **50.89** | 68.94 | 66.03 | 67.45 |
| Full supervised segmentation, epoch 5 | DINO + metric depth | **45.94** | 67.20 | 59.21 | 62.95 |

Depth-supervision deltas are **+4.86 pp** at the camera probe, **+4.52 pp** at
the fused probe, and **-4.95 pp** after full supervised adaptation.

### Final epoch versus best validation epoch

| Readout | DINO-only best IoU / epoch | Depth-supervised best IoU / epoch |
| --- | ---: | ---: |
| Camera probe | 0.0076% / 10 | 4.90% / 9 |
| Fused probe | 7.60% / 8 | 12.92% / 4 |
| Full segmentation | 50.89% / 5 | 47.29% / 4 |

The historical trained-readout comparison uses the preset final epochs consistently. Even comparing
best validation epochs, the depth-supervised full segmentation remains lower
by **3.60 pp**. Best-epoch selection uses the validation set; it is not an
independent test estimate. The preset budgets were not changed after inspection.

## What the results mean

### Frozen representations: improvement, with a severe camera-probe problem

Adding depth supervision increased frozen semantic readability in both probes.
Across 150 scenes, the camera probe improved on 148 scenes and tied on two;
the fused probe improved on 128, worsened on 19 and tied on three. These are
paired scene diagnostics under the same seed and protocol, not repeated-seed
statistical significance.

Absolute probe performance is still low. The DINO-only camera probe produced
only **286 true-positive vehicle cells**, against **3,747,611 GT-positive cells**
in the valid evaluation region. It labelled just **0.0132%** of valid cells as
vehicles, whereas GT vehicles occupy **1.5650%**. The depth-supervised camera
probe labelled **0.3462%** as vehicles. This directly demonstrates extreme
underprediction, particularly for the control.

The frozen result measures what this particular linear head can read under its
fixed optimization and threshold. It does not establish that every useful
semantic feature is absent. Feature magnitude/normalization, the lifting and
height-reduction readout, probe optimization, logits and class imbalance need
inspection before attributing the near-zero result solely to failed semantics.
No threshold sweep, feature-normalization intervention or linear-head debugging
experiment was performed for this report.

Within each arm, post-fusion probe IoU is higher than pre-fusion probe IoU:
**0.0076 -> 7.43%** for control and **4.87 -> 11.95%** for treatment. That supports
greater semantic readability at the fused readout in this setup. It does not
isolate radar causality: fusion processing, supervision location and feature
conditioning also change. Both arms use radar and velocity; neither provides
a matched camera-only training control or a velocity-hidden training control.

### End-to-end adaptation: control is currently stronger

The control's five validation IoUs increased steadily:
**44.74 -> 48.27 -> 49.65 -> 50.45 -> 50.89%**.
The treatment's curve was less stable:
**41.88 -> 46.21 -> 46.00 -> 47.29 -> 45.94%**.

At the final epoch, depth supervision reduces recall by **6.82 pp** and
precision by **1.74 pp**. It yields **255,429 fewer true-positive cells** and
255,429 more false-negative cells, while removing only 31,765 false positives.
The loss in IoU is therefore mainly associated with more missed vehicle cells.
Full segmentation is worse on **126 of 150 scenes**, and better on 24.
The scene-level IoU difference has median **-3.57 pp**. This scene result is
separate from the globally pooled **-4.95 pp** primary IoU difference.

The full decoder and GT adaptation substantially improve both models over their
frozen linear probes. This establishes usable supervised adaptation in the
tested budget, not that self-supervised pretraining alone explains the result.
A matched supervised-from-scratch or no-DINO training control is still missing.

### Pretraining diagnostics: added depth does not track downstream IoU directly

| Validation term | DINO-only, epoch 1 -> 50 | Depth-supervised, epoch 1 -> 50 |
| --- | ---: | ---: |
| Image DINO loss | 0.3221 -> 0.2084 | 0.3512 -> 0.2231 |
| Fused-BEV DINO loss | 0.2320 -> 0.1733 | 0.2607 -> 0.3540 |
| Metric-depth CE | Not active | 2.8271 -> 3.3554 |

The treatment's held-out BEV and depth objectives worsen over the full run,
despite improved frozen-probe IoU. This is a real discrepancy between proxy
objectives and downstream readouts, not evidence that one loss is an IoU.
The two total losses cannot be compared directly because the treatment has an
extra term. The depth numbers are agreement with frozen pseudo-depth, not
metric depth MAE in metres or independent dense-depth GT accuracy. The archived
depth MAE/RMSE fields are null and radar-anchor counts are zero for this profile.

Possible explanations include multi-objective optimization tradeoffs,
pseudo-depth transfer/calibration and readout conditioning. They remain
hypotheses; the present measurements do not identify the mechanism or prove
that Depth Anything is inherently harmful.

## Relation to the official Simple-BEV reference

The separate checkpoint-reproduction report at
`/home/students2026/dd2414-baseline/simple_bev/REPRODUCTION_REPORT.md` records
**47.5580%** camera-only and **55.8120%** camera+radar Vehicle IoU at 448x800.
The current control is **4.92 pp below** the latter reference.

This is a contextual reference, not a matched training-method ablation. The
official radar reference uses five sweeps, different encoder/radar architecture,
initialization and training budget; its reproduced evaluator scores **6,016**
frames because of `drop_last=True`. This study scores all **6,019** frames.
Do not call the current 50.89% a radar gain over official camera-only, or claim
that DINO alone caused the gap to the official camera+radar checkpoint.

## Evidence, completion and reproducibility

- All two pretrains plus six downstream jobs are `COMPLETED` with exit `0:0`.
  DINO-only jobs: 1251176-1251179; treatment jobs: 1251180-1251183.
- Each pretrain completed exactly 50 epochs / 281,300 updates. Frozen probes
  completed 10 epochs; full segmentation completed five. All eight final
  summaries and every recorded epoch validation cover 6,019 samples.
- Every final summary/config was read from the durable backup and its SHA-256
  checked against the backup manifest. For all six final GT evaluations,
  6,019 archived per-frame records reproduce the reported confusion counts
  exactly and span the same 150 scenes. IoU/Recall were also recomputed from
  final counts. Both arms record the same initial student hash and protocol.
- The original trainer is immutable commit
  `a31f9039f9fe3e265a503a1ffdedc03b71e361a0`; admitted runtime controller is
  `aec677657cd0a72b7aa1ef72510faf4f1d3bee2a`; fixed continuation helper is
  `4a4fbdca34597f2ae7c76044314306bc3606b69d`. All three execution worktrees
  remain clean at the live check. Later source refactoring was not used to
  generate these results.
- Durable root:
  `/Midgard/home/hongen/projects/DD2414-Project/fullsize_baseline/runs/random_resnet101_metric_bev_50epochs_workers0_resume_20261109`.
  Inspect `<arm>/<readout>/summary.json`, `config.json`,
  `validation/epochNNN/summary.json`, `per_frame.jsonl` and the checkpoint tree.
  GT readout directory names match the three table protocols.
- Backup job **1251184 COMPLETED**, with a `verified` manifest listing **543
  files**. The cleanup record confirms removal of the authorized eowyn dataset
  copy after backup, freeing approximately **396.72 GiB**. Run/checkpoint
  outputs and workstation data were preserved. No data were deleted by this
  reporting task. The current audit freshly rechecked report-input hashes,
  not every large checkpoint in the 543-file archive.

To reproduce the primary result table without rerunning a model, on moria use
the existing Miniforge interpreter for these eight small JSON files:

```bash
/Midgard/home/hongen/miniforge3/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/Midgard/home/hongen/projects/DD2414-Project/fullsize_baseline/runs/random_resnet101_metric_bev_50epochs_workers0_resume_20261109')
manifest = json.loads((root / 'backup_manifest.json').read_text())['sha256']
for arm in ('dino-only', 'dino-metric-depth-bev'):
    for mode in ('pretrain', 'frozen_camera_bev_probe',
                 'frozen_fused_bev_probe', 'end_to_end_segmentation'):
        path = root / arm / mode / 'summary.json'
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest[str(path.relative_to(root))]
        summary = json.loads(content)
        assert summary['status'] == 'complete'
        assert summary['final_validation']['samples'] == 6019
        print(arm, mode, summary['final_validation'])
PY
```

Large-file/checkpoint rehashes, per-frame analyses and model inference belong
in explicit CPU/GPU Slurm allocations, not on the login node. Archived native
mask panels were copied unchanged for the local report and hash-verified;
there was no new model inference, threshold tuning or training in this review.

## Recommended next steps

1. Audit the nearly background-only camera probe: verify the selected tensor,
   spatial alignment, probability-weighted feature magnitude and logit/gradient
   distributions, using the existing checkpoints before launching a long run.
2. Retain 50.89% as the stronger current full-segmentation baseline. Investigate
   the depth treatment's worse full-scene recall and late validation degradation.
   Use training-side experiments for any loss-weight/normalization change;
   preserve the present fixed-threshold scores as the original result.
3. Add a matched no-DINO or supervised-from-scratch control before attributing
   improvement to distillation. Add matched radar/velocity and sweep controls
   only if making claims about radar, motion or ten-sweep benefit.
4. Repeat the selected configuration with additional seeds. The present paired
   scene results show breadth, but do not measure training-seed uncertainty.

No additional experiments are launched by this report.
