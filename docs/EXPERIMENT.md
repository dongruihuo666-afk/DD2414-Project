# Experiment: `experiment/rpl-depth-10sweep-velocity-study`

This branch contains the ten-sweep depth and velocity studies, including the 50-epoch experiment that trains ResNet101 from scratch.

Original source: `0f91e222d658439d253dde4babf9a353056eecd3`.
The corresponding source tag is listed in [reproducibility](REPRODUCIBILITY.md).

## Model and training

`MultiSweepTeacher` supports image DINO distillation and radar-supported BEV
DINO distillation, with the motion and depth profiles implemented in this
branch. Supplying `--official-camera-checkpoint` strictly loads `encoder.*`
from the official Simple-BEV RGB checkpoint. Otherwise the factory uses its
original initialization.

The `random-metric-bev` profile trains for 50 total epochs with ten radar sweeps,
velocity input, 448x800 RGB, BF16 and a per-rank batch size of one. Both arms
use depth probabilities for lifting. The treatment adds metric Depth Anything
supervision after projection into the camera BEV volume and before radar fusion.
Its validator fixes the loss weights, rank counts and other protocol settings.
Historical continuation helpers restore only their admitted parent runs.

## Evaluation

Frozen camera/fused probes measure representation quality under a fixed linear
readout. Unfrozen readouts and supervised segmentation measure adaptation with
labels. GT-box velocity readouts evaluate motion at known object locations.
Report these protocols separately.

The owner now prioritizes standard Simple-BEV segmentation metrics over frozen
linear probes. The historical probe results remain recorded, but are not the
primary assessment for new reporting. Direct inference of the completed
pretraining checkpoint is a separate protocol: use the archived segmentation
decoder without fitting a probe or updating any parameter. That decoder was
excluded from pretraining, so these scores must be labelled as an untrained
readout; they do not replace supervised fine-tuning results without explanation.

`tools/evaluate_pretrained_checkpoint.py` implements this read-only evaluation
for the original fifty-epoch random-metric-BEV study. It requires an immutable
`a31f9039f9fe` checkout and the archive's exact checkpoint SHA-256, validates
source/config/completion, loads every tensor strictly, and checks that the
decoder equals its original random initialization. It evaluates all 6,019
validation frames, using the original valid-cell sigmoid/round confusion counts,
and verifies unchanged parameters/BN buffers before writing a complete result.
Default inference is FP32 with TF32 disabled; no teacher or optimizer is created.
An explicit `--pilot-only` checks one runtime batch without reporting a score.

```bash
python tools/evaluate_pretrained_checkpoint.py \
  --source-root /ABSOLUTE/PATH/TO/IMMUTABLE_A31_SOURCE \
  --checkpoint /ABSOLUTE/PATH/TO/PRETRAIN_CHECKPOINT.pt \
  --sha256 ARCHIVED_CHECKPOINT_SHA256 \
  --treatment dino-only \
  --data-root /ABSOLUTE/PATH/TO/nuscenes \
  --bevcar-source /ABSOLUTE/PATH/TO/BEVCar \
  --run-dir /ABSOLUTE/PATH/TO/FRESH_OUTPUT \
  --batch-size 4 --num-workers 2 --precision float32
```

Repeat with `--treatment dino-metric-depth-bev` and its own checkpoint/hash/output.
The two checkpoints and their hashes are recorded in the completed-results report.

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
