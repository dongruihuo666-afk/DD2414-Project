# Hybrid-teacher 10-sweep cluster experiment

This opt-in path leaves the historical Simple-BEV and full-size baseline
entrypoints unchanged. It implements the agreed complete model and two
downstream evaluations from one common self-supervised checkpoint.

## What the complete model contains

The trainable student combines the Simple-BEV ResNet-101 camera encoder, the
official BEVCar `nets/voxelnet.py` radar encoder, ten radar sweeps, the existing
camera-to-BEV geometry, camera/radar BEV fusion, and the original Simple-BEV
decoder. A frozen DINOv2 teacher supplies exactly two pretraining losses:

1. dense image-feature distillation on the trainable camera encoder; and
2. fused-BEV DINO distillation weighted by radar-covered soft regions.

Their weighted sum is the hybrid-teacher objective. The radar motion loss is a
separately weighted auxiliary pretraining term, not a third teacher. DINOv2 is
never updated and is not part of the deployable student.

After pretraining, the same checkpoint is evaluated in two independent runs:

- **All student modules trainable:** the DINO and motion losses are disabled;
  the complete student is optimized with the original Simple-BEV segmentation,
  center and offset losses.
- **Frozen representation plus 1x1 probe:** every pretrained model parameter is
  frozen and only a one-channel `Conv2d(128, 1, kernel_size=1)` is optimized.

The primary metric in both runs is the original Simple-BEV mean-per-frame
Vehicle BEV IoU on the complete 6,019-frame nuScenes validation split. Global
IoU, precision, recall and F1 are written only as auxiliary diagnostics.

## Cluster setup

Clone and select the experiment branch, create the same Python environment as
the base repository, stage nuScenes `v1.0-trainval`, and separately obtain the
pinned BEVCar source (commit `29cacda3`). DINOv2 weights must already be in the
PyTorch Hub cache on an offline compute node, or the node must be allowed to
download them once.

Set paths explicitly:

```bash
export PYTHON_BIN=/path/to/env/bin/python
export NUSCENES_ROOT=/path/to/nuscenes
export BEVCAR_SOURCE_DIR=/path/to/BEVCar
export OUTPUT_ROOT=/persistent/path/hybrid_teacher_cluster_runs
```

Check imports, CUDA, data metadata, BEVCar, and syntax without training:

```bash
bash cluster_experiment/train.sh check
```

Run the real ten-sweep smoke protocol before requesting a long allocation:

```bash
bash cluster_experiment/train.sh smoke
```

Run all production stages sequentially with automatic checkpoint resume:

```bash
bash cluster_experiment/train.sh all
```

Stages can also be launched separately with `pretrain`, `finetune`,
`linear-probe`, or `evaluate`. Progress is available in each run directory as
`console.log`, `status.json`, `metrics.jsonl`, TensorBoard events, epoch
validation JSON, and atomic checkpoints. `SIGUSR1` and `SIGTERM` request a safe
checkpoint after the current optimizer update.

For Slurm, first obtain the real partition/account/GPU rules from the target
cluster. Then set `SLURM_PARTITION` and, if required, `SLURM_ACCOUNT`; optional
resource variables are `SLURM_GPUS`, `SLURM_CPUS`, `SLURM_MEMORY`, and
`SLURM_TIME`:

```bash
export SLURM_PARTITION=<verified-partition>
export SLURM_ACCOUNT=<verified-account>
bash cluster_experiment/submit_slurm.sh all
```

No partition or account is guessed by the repository. The default physical
batch size is two because the combined trainable camera path is heavier than
the prior frozen-camera baseline. Increase it only after the real smoke run
shows sufficient memory headroom; use gradient accumulation to change the
effective batch size without changing physical memory demand.

## Comparison boundary

The evaluator now uses the same Vehicle BEV IoU formula and supervised loss
components as Simple-BEV. A numerical comparison to a published or checkpoint
baseline is defensible only when input resolution, train/validation split,
training schedule, augmentation, and evaluation code are also matched. The
generated `comparison/official_metrics.md` states this boundary explicitly.
