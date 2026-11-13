# This experiment: `experiment/rpl-full-data-double-distill`

Independent official-RGB full-data double-distillation deployment and downstream protocol.

The pre-cleanup source is `f9f333f8607f22514df216d5969f23a0ecbf5917`, saved under
`repository-snapshot-20261113T185518Z/experiment/rpl-full-data-double-distill`. The model/data implementation, loss weights, initialization,
velocity coordinate transforms, sweeps, depth geometry and budget are preserved
from that source. Engineering extraction does not add a new experiment or result.

## Configuration and evaluation

`MultiSweepTeacher` exposes existing image DINO and radar-supported BEV DINO
objectives plus its existing motion/depth profiles. Frozen camera/fused probes,
unfrozen readouts, full supervised segmentation and GT-box motion readouts are
different protocols and must be reported separately. The official-initialized
protocol uses only strict `encoder.*` weights from the official RGB checkpoint;
absence of that explicit argument retains the branch's original initialization.

Full-data results use the official train/validation scene separation in the
branch's recorded manifest. Public test labels are withheld; validation scores
are not a new blind-test result. An older subset protocol in an older branch is
not silently regenerated, relabeled as full data or removed as a method change.
Use existing manifest validation and exact token order/hash.

## Recommended entrypoints

```bash
dd2414 train --help
```


All flags/defaults come verbatim from this branch's original parser. Existing
`scripts/train_*.py` commands remain available. The original official
`train_nuscenes.py` / `eval_nuscenes.py`, download scripts and checkpoint keys are
unchanged. The upstream `eval.sh` defaults are not a universal reproduction recipe.

For exact historical commands/configuration/status snapshots, recover original
documents from the tag rather than treating old RUNNING statements as live
facts. No cluster query, new job or full-data retraining is part of this cleanup.
Small curated evidence is indexed in [results](../results/README.md).
