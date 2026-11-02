# Full-data experiment admission and test provenance

The owner requested removal of subset experiments and an automatic high-resolution
Simple-BEV RGB encoder warm-start study. The official checkpoint baselines are
already complete. Automatic training is **blocked by the unresolved test
protocol**: the requested combination of a truly unseen labeled local test set
and the existing supervised encoder cannot be obtained by merely renaming splits.

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

## Decision needed before scheduling

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

Do not automatically choose one of these options. Do not train on test inputs
for self-supervision, normalize using test statistics, fit probes on held-out GT,
or select epochs/thresholds based on final test scores. Future splits must keep
sample, scene, log, and multi-sweep-history ownership disjoint, and bind sampler,
checkpoint, cache and report provenance to the approved manifest hash.

The inactive candidate for option 1 keeps all 700 train scenes / 28,130 frames
and divides official val into 75 validation scenes / 3,009 frames / nine logs
and 75 final-holdout scenes / 3,010 frames / nine different logs. Assignment is
seeded and uses metadata only. It is not activated or represented as blind test.

## Concrete scope awaiting that decision

The audit emits eleven pending full-data tasks: one initial frozen camera-BEV
probe; two five-epoch double-DINO models (velocity supplied vs hidden throughout
training); four ten-epoch before/after-fusion frozen probes; two five-epoch
GT-vehicle velocity readouts; two five-epoch segmentation fine-tunes. RGB input is
448x800, radar is ten sweeps, and the official RGB encoder is strictly loaded.
These task records are a specification, **not a running or executable queue**.
The high-resolution trainer, probe locations, guards and persistent scheduling
remain to be implemented after the test protocol is selected.

Reproduce the audit:

```bash
PYTHONPATH=scripts MPLCONFIGDIR=$PWD/.cache/matplotlib \
  $HOME/miniconda3/envs/bev/bin/python scripts/audit_full_data_protocol.py \
  --output artifacts/full_data_protocol_audit.json --require-ready
```

Exit code 2 means the unresolved protocol blocks automatic launch. Historical
scaling results stay in Git as evidence; numeric scale generation, prefix
selection and the 4,096-sample launcher branch have been removed. The active
manifest contains only `full` and its hash changed. Old checkpoint resumes must
use their pinned historical code/manifest, not bypass a hash mismatch.
