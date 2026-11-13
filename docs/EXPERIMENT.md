# This experiment: `experiment/hybrid-teacher-10sweep-cluster`

Independent historical full hybrid-teacher experiment, its own model/readout and RPL environment protocol.

The pre-cleanup source is `33f2d86d70286c514cede6539f991825dde726ee`, saved under
`repository-snapshot-20261113T185518Z/experiment/hybrid-teacher-10sweep-cluster`. The model/data implementation, loss weights, initialization,
velocity coordinate transforms, sweeps, depth geometry and budget are preserved
from that source. Engineering extraction does not add a new experiment or result.

## Configuration and evaluation

The legacy hybrid path uses `HybridTeacherModel`, its own heads and original
initialization flag; do not substitute the later metric-depth model. Its
historical manifest/scale selectors are retained exactly for reproduction.
The frozen probe, supervised adaptation and Simple-BEV mean-per-frame metric
remain separate from auxiliary global confusion-count metrics.

Full-data results use the official train/validation scene separation in the
branch's recorded manifest. Public test labels are withheld; validation scores
are not a new blind-test result. An older subset protocol in an older branch is
not silently regenerated, relabeled as full data or removed as a method change.
Use existing manifest validation and exact token order/hash.

## Recommended entrypoints

```bash
dd2414 downstream --help
```

```bash
dd2414 pretrain --help
```


All flags/defaults come verbatim from this branch's original parser. Existing
`scripts/train_*.py` commands remain available. The original official
`train_nuscenes.py` / `eval_nuscenes.py`, download scripts and checkpoint keys are
unchanged. The upstream `eval.sh` defaults are not a universal reproduction recipe.

For exact historical commands/configuration/status snapshots, recover original
documents from the tag rather than treating old RUNNING statements as live
facts. No cluster query, new job or full-data retraining is part of this cleanup.
Small curated evidence is indexed in [results](../results/README.md).
