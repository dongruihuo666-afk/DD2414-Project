# Full-data experiment admission and test provenance

## Approved protocol (2026-11-03)

The owner explicitly chose the original official Simple-BEV split: all 700 train
scenes / 28,130 frames for all learning, and all 150 val scenes / 6,019 frames
for validation. No additional holdout is created. The baseline checkpoints have
already been reproduced. Results from the new study are **validation results**,
not newly untouched blind-test scores. Previous validation evaluation does not
mean that those frames were used for gradient training. The earlier protocol
decision is resolved and no longer blocks training admission.

`configs/full_data_official_protocol.json` freezes the approved configuration.
`scripts/run_official_full_data.py` is the executable persistent queue, launched
by `bash scripts/start_official_full_data.sh`. It waits for other GPU clients
to finish without stopping them. Actual runtime state is in
`fullsize_baseline/runs/official_rgb_full_10sweep_20261103/pipeline.json`;
do not infer running training from this document or a ready audit alone.

## Verified dataset facts

- Official trainval: 700 train scenes / 28,130 frames; 150 validation scenes /
  6,019 frames. Active manifests and scale selectors now support full only.
- Downloaded official test metadata has 150 scenes / 6,008 frames, but its
  `sample_annotation.json` and `instance.json` each contain zero records.
- Public test metadata source: `https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/v1.0-test_meta.tgz`.
  SHA-256: `c5f5169875eead4874fc345235e3fc9a4b60dcad42a18386a824a1d1d58f3ed6`.
  The archive is cached locally under `.cache/nuscenes/`; it is not in Git.
- The [official evaluation rules](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md)
  confirm that test annotations are withheld. The public detection/tracking
  benchmarks are not a local scorer for our binary BEV mask/probe outputs.
- The requested official RGB encoder was trained on the official 700-scene
  training set. A later split cannot erase that initialization exposure.
- All 6,019 official validation frames were already scored during project
  development. They are held out from gradient training, but cannot now be
  represented as a newly untouched blind test set.

No official test sensor blobs have been downloaded: they do not supply the
missing GT needed for Vehicle BEV IoU and GT-box-conditioned velocity EPE.
Downloading metadata verified this directly before spending storage on blobs.

## Earlier alternatives (historical, not selected)

Possible protocols have different scientific claims:

1. Keep the requested RGB checkpoint, train on official train, and split official
   val by complete logs into validation and final holdout. This prevents gradient
   and temporal-history overlap, but the holdout is historically observed. It
   must be reported with that limitation; it is not a pristine blind test.
2. Obtain independent compatible labeled data, or arrange annotations for official
   test. This preserves the requested warm-start but requires new data/annotation
   authority and work beyond downloading public sensor files.
3. Change initialization to a model not trained on nuScenes and create a new
   log-isolated train/validation/test protocol. This changes the explicitly
   requested camera initialization and needs the owner's decision. Historical
   experiment exposure still needs a documented assessment.

The owner instead chose unchanged official train/val membership. Do not train on test inputs
for self-supervision, normalize using test statistics, fit probes on held-out GT,
or select epochs/thresholds based on final test scores. Future splits must keep
sample, scene, log, and multi-sweep-history ownership disjoint, and bind sampler,
checkpoint, cache and report provenance to the approved manifest hash.

The inactive candidate for option 1 keeps all 700 train scenes / 28,130 frames
and divides official val into 75 validation scenes / 3,009 frames / nine logs
and 75 final-holdout scenes / 3,010 frames / nine different logs. Assignment is
seeded and uses metadata only. It is not activated or represented as blind test.

## Executable experiment scope

The queue executes eleven full-data tasks: one initial frozen camera-BEV
probe; two five-epoch double-DINO models (velocity supplied vs hidden throughout
training); four ten-epoch before/after-fusion frozen probes; two five-epoch
GT-vehicle velocity readouts; two five-epoch segmentation fine-tunes. RGB input is
448x800, radar is ten sweeps, and the official RGB encoder is strictly loaded.
The camera encoder loads **every `encoder.*` tensor** from the official RGB
checkpoint, with strict key/shape validation and recorded SHA-256. Other modules
are newly initialized; no official decoder or fusion weights are imported.
DINO remains frozen. Its input is 448x798, the nearest patch-14-compatible width
to 800. Camera input is 448x800. Both velocity treatments use the same full split,
budget and initialization; the hidden treatment zeroes only VoxelNet velocity
input channels during all training and evaluation. Radar-derived auxiliary
velocity targets are not zeroed.

Warm-start datasets explicitly use checkpoint-era Pillow RGB default resizing,
as in the successful official reproduction, instead of the experimental legacy
nearest-neighbor resize. This is an opt-in dataset configuration; existing
Simple-BEV/default transforms are unchanged. All 6,019 validation frames are
scored without dropping the final batch. The exact earlier official reproduction
scored 6,016 because its loader dropped the last batch; report that small coverage
difference when citing its published/reproduced score. No inference-resolution
downscaling or subset selection is applied in this queue.

Both frozen probes use one `Conv2d(128,1,1)` and identical ten-epoch training.
Before fusion, fixed mean pooling over the eight height bins reduces the
1024-channel camera volume to 128 channels. After fusion, the existing learned
fusion compressor produces 128 channels. This is a location diagnostic, **not
a perfect causal radar ablation**, because compression also differs. The
velocity-present/hidden training control and radar-input interventions provide
additional context. A supervised camera-only matched re-training control is not
part of these eleven tasks. GT object motion readouts use known vehicle boxes;
they do not measure whole-scene detection or nuScenes NDS.

The queue first performs two real full-manifest pretraining updates and resumes
in a new process to update three. These are engineering checks, not subset
experiments. It checks finite loss/gradient norms before admitting the formal
tasks. Batch size is conservatively one on the 24-GiB card; BF16, eight workers,
periodic atomic checkpoints, full validation after every epoch, TensorBoard and
incremental progress files are enabled. A real high-resolution GPU check has not
yet passed while another job occupies the GPU. If it fails, the queue stops and
records the failure rather than mutating the scientific configuration.

There are 80 model-training epochs across the eleven tasks, plus full validations
and engineering checks. This is not an overnight completion guarantee. The
single-seed five-epoch pretraining budget is not a convergence claim. Only
Vehicle IoU matches the principal Simple-BEV segmentation metric; precision,
recall and F1 are supplementary, and reconstruction losses are
diagnostics. Threshold remains fixed at sigmoid.round(), never fit on val.

Reproduce the audit:

```bash
PYTHONPATH=scripts MPLCONFIGDIR=$PWD/.cache/matplotlib \
  $HOME/miniconda3/envs/bev/bin/python scripts/audit_full_data_protocol.py \
  --protocol official_train_val \
  --output artifacts/full_data_protocol_audit.json --require-ready
```

Exit code 0 admits the chosen official train/val protocol; it does not certify
GPU training or completed experiments. Omitting the protocol flag retains the
old fail-closed decision mode (exit code 2 with `--require-ready`). Historical
scaling results stay in Git as evidence; numeric scale generation, prefix
selection and the 4,096-sample launcher branch have been removed. The active
manifest contains only `full` and its hash changed. Old checkpoint resumes must
use their pinned historical code/manifest, not bypass a hash mismatch.

Monitor without loading a model:

```bash
watch -n 10 "$HOME/miniconda3/envs/bev/bin/python scripts/show_official_full_data.py"
tail -F fullsize_baseline/runs/official_rgb_full_10sweep_20261103/queue.log
systemctl --user status dd2414-official-full-data.service
```

Stop safely with `systemctl --user stop dd2414-official-full-data.service`.
The child saves at its next update boundary. Restart the launcher to continue.
Unrelated GPU jobs are never signaled. The GPU idle gate cannot reserve the
device against another user starting a new job later; this is a local queue,
not a Slurm resource allocation. Runtime checkpoints/logs stay outside Git.
After each formal stage, the queue attempts a scoped handoff/results commit,
push and remote-ref verification. Manual edits to those exact report paths
defer publication; unrelated staged files are never included. Publication
failure is logged and does not discard training or stop the next task.
