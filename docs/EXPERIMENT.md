# Experiment: `experiment/rpl-full-data-double-distill`

This branch contains the official-RGB-initialized double-distillation experiment and its full-data RPL deployment.

Original source: `f9f333f8607f22514df216d5969f23a0ecbf5917`.
The corresponding source tag is listed in [reproducibility](REPRODUCIBILITY.md).

## Model and training

`MultiSweepTeacher` supports image DINO distillation and radar-supported BEV
DINO distillation, with the motion and depth profiles implemented in this
branch. Supplying `--official-camera-checkpoint` strictly loads `encoder.*`
from the official Simple-BEV RGB checkpoint. Otherwise the factory uses its
original initialization.

## Evaluation

Frozen camera/fused probes measure representation quality under a fixed linear
readout. Unfrozen readouts and supervised segmentation measure adaptation with
labels. GT-box velocity readouts evaluate motion at known object locations.
Report these protocols separately.

Full-data results use the official train/validation scene split recorded in
the manifest. Public test annotations are withheld, so these are validation
scores. Keep the recorded sample order and manifest hashes when reproducing a run.

## Commands

```bash
dd2414 train --help
```


The existing `scripts/train_*.py` commands remain available. Official Simple-BEV
uses `train_nuscenes.py`, `eval_nuscenes.py` and its download scripts. Match
their preprocessing and arguments to the checkpoint being evaluated.

[Results](../results/README.md) indexes the retained benchmark summaries.
Original commands, detailed configurations and reports can be recovered from
the source tag. Check running jobs directly before using an old status record.
