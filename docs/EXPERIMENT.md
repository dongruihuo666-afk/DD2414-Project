# Experiment: `experiment/hybrid-teacher-10sweep-cluster`

This branch implements the original hybrid-teacher experiment and its RPL cluster setup.

Original source: `33f2d86d70286c514cede6539f991825dde726ee`.
The corresponding source tag is listed in [reproducibility](REPRODUCIBILITY.md).

## Model and training

`HybridTeacherModel` combines the camera encoder, BEVCar radar encoder and
image/BEV distillation heads. It has separate pretraining and downstream
commands. The original initialization flags, heads and manifest selectors are
retained for this experiment.

## Evaluation

A frozen 1x1 probe reads the fused BEV representation. End-to-end segmentation
trains the student with ground-truth labels. These are the two downstream modes
implemented in this branch.

Full-data results use the official train/validation scene split recorded in
the manifest. Public test annotations are withheld, so these are validation
scores. Keep the recorded sample order and manifest hashes when reproducing a run.

### Historical IoU naming issue

`scripts/train_hybrid_teacher_downstream.py` labels the mean of per-frame IoUs
as `official_simplebev.vehicle_bev_iou`. The original `eval_nuscenes.py` instead
sums intersection and union over the validation set before dividing. These
aggregations generally differ. The historical global-confusion result also uses
`>=0.5`, whereas official inference uses `sigmoid().round()`, which differs at
exactly zero logits. Treat the old field as mean per-frame IoU, not a matched
official score. Historical numbers and field names are retained pending a
separate correction and re-evaluation.

## Commands

```bash
dd2414 downstream --help
```

```bash
dd2414 pretrain --help
```


The existing `scripts/train_*.py` commands remain available. Official Simple-BEV
uses `train_nuscenes.py`, `eval_nuscenes.py` and its download scripts. Match
their preprocessing and arguments to the checkpoint being evaluated.

[Results](../results/README.md) indexes the retained benchmark summaries.
Original commands, detailed configurations and reports can be recovered from
the source tag. Check running jobs directly before using an old status record.
