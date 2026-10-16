# Hybrid Teacher 10-sweep experiment on KTH RPL

This is an opt-in cluster path. It does not change the historical Simple-BEV
entrypoints, checkpoint format, model structure, loss definitions, or training
logic. It prepares one pinned Python environment and launches the existing
Hybrid Teacher protocol through Slurm.

Never train on the `moria` login node. Environment installation and the CPU
audit may run there; CUDA verification, smoke tests, and training must run in a
Slurm GPU allocation.

## Experiment represented by this branch

The trainable student combines the Simple-BEV ResNet-101 camera encoder, the
official BEVCar `nets/voxelnet.py` radar encoder, ten radar sweeps, the existing
camera-to-BEV geometry, camera/radar BEV fusion, and the original Simple-BEV
decoder. A frozen DINOv2 teacher supplies two pretraining losses:

1. dense image-feature distillation on the trainable camera encoder; and
2. fused-BEV DINO distillation weighted by radar-covered soft regions.

Their weighted sum is the Hybrid Teacher objective. Radar motion loss is a
separately weighted auxiliary term, not another teacher. DINOv2 is frozen and
is not part of the deployed student.

One pretraining checkpoint feeds two independent downstream experiments:

- all student modules trainable, optimized with the original Simple-BEV
  segmentation, center, and offset losses;
- the pretrained representation frozen, with only a one-channel
  `Conv2d(128, 1, kernel_size=1)` linear probe trained.

The primary metric is Simple-BEV's mean-per-frame Vehicle BEV IoU on the full
6,019-frame nuScenes validation split. Global IoU, precision, recall, and F1
remain auxiliary diagnostics.

## Dependency boundary

The source import audit for this path found these runtime groups:

- PyTorch 2.5.1 and TorchVision 0.20.1 from the official CUDA 12.4 wheel index;
- NumPy/SciPy/scikit-learn/scikit-image, fixed around NumPy 1.26.4;
- nuscenes-devkit 1.2.0 plus its image, geometry, and plotting dependencies;
- efficientnet-pytorch, Fire, TensorBoard, and tensorboardX, which are imported
  by unchanged Simple-BEV modules or training utilities;
- one pinned BEVCar source file and one pinned DINOv2 source/weight cache.

The full BEVCar training framework is not required. In particular, this path
does not install Lyft SDK, MMCV, MMDetection3D, spconv, or xFormers. The pinned
BEVCar VoxelNet imports only PyTorch. DINOv2's memory-efficient xFormers path is
optional for this backbone; its PyTorch fallback is used when xFormers is not
installed. Ordinary dependencies are listed in `requirements-rpl.txt`; torch
and torchvision are deliberately installed in a separate phase so a CPU-only
wheel cannot be selected accidentally.

## 1. Clone and install the environment on moria

The documented project and Miniforge locations are:

```bash
cd /Midgard/home/hongen/projects/DD2414-Project
git switch experiment/hybrid-teacher-10sweep-cluster
git pull --ff-only
bash cluster_experiment/setup_rpl_env.sh
```

The installer creates `dd2414-hybrid` with Python 3.10 only when it does not
already exist. If it exists, the script verifies its Python version and updates
the pinned packages without deleting the environment. It then runs `pip check`,
the project import audit, and prints the exact `PYTHON_BIN` export command.

Expected default:

```bash
export PYTHON_BIN=/Midgard/home/hongen/miniforge3/envs/dd2414-hybrid/bin/python
```

The installer refuses source builds except for `efficientnet-pytorch` 0.7.1,
whose upstream release is a small pure-Python source distribution with no
native compiler step. It therefore cannot silently begin a CUDA/C++ build on
`moria`. A missing binary wheel for any other package is an error that should
be resolved in a compute allocation or by revising the pin; do not hide it
with `--no-deps`.

To audit an existing environment without changing it:

```bash
export BEVCAR_SOURCE_DIR=/Midgard/home/hongen/src/BEVCar
export TORCH_HOME=/Midgard/home/hongen/.cache/torch
MPLBACKEND=Agg "$PYTHON_BIN" cluster_experiment/verify_rpl_env.py \
  --mode cpu --bevcar-source "$BEVCAR_SOURCE_DIR"
```

CPU mode checks package versions, imports, the CUDA-enabled torch build,
BEVCar, and the DINOv2 cache. It explicitly reports CUDA as *not tested*.

## 2. Install the pinned BEVCar VoxelNet source

Only `nets/voxelnet.py` is loaded; do not install BEVCar's old environment.

```bash
mkdir -p /Midgard/home/hongen/src
git clone https://github.com/aharley/BEVCar.git /Midgard/home/hongen/src/BEVCar
git -C /Midgard/home/hongen/src/BEVCar checkout \
  29cacda3bc5416d47428c1d0f017527acad34f90
export BEVCAR_SOURCE_DIR=/Midgard/home/hongen/src/BEVCar
sha256sum "$BEVCAR_SOURCE_DIR/nets/voxelnet.py"
```

The expected VoxelNet SHA-256 is
`43ed0ee30f70c328bd47885cbc68272e7f5e3625c61b325b54f713a851f3e6ac`.
The verifier checks both the file digest and, for a Git checkout, the commit.

## 3. Prepare the pinned DINOv2 cache

Compute nodes may not have Internet access, so prepare the shared PyTorch Hub
cache before submitting. All nodes see `/Midgard/home`.

```bash
export TORCH_HOME=/Midgard/home/hongen/.cache/torch
mkdir -p "$TORCH_HOME/hub/checkpoints"
git clone https://github.com/facebookresearch/dinov2.git \
  "$TORCH_HOME/hub/facebookresearch_dinov2_main"
git -C "$TORCH_HOME/hub/facebookresearch_dinov2_main" checkout \
  7764ea0f912e53c92e82eb78a2a1631e92725fc8
wget -O "$TORCH_HOME/hub/checkpoints/dinov2_vits14_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
sha256sum "$TORCH_HOME/hub/checkpoints/dinov2_vits14_pretrain.pth"
```

The expected weight SHA-256 is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
If either directory already exists, inspect it and fetch/check out the pinned
revision rather than cloning over it.

## 4. Set data and output locations

The nuScenes location on RPL is not known by this repository. Find or stage the
official `v1.0-trainval` data first, then point to the directory containing both
the metadata directory and `samples`, `sweeps`, and `maps`:

```bash
export NUSCENES_ROOT=<verified-shared-path-to-nuscenes>
test -d "$NUSCENES_ROOT/v1.0-trainval"
test -d "$NUSCENES_ROOT/samples"
test -d "$NUSCENES_ROOT/sweeps"

export OUTPUT_ROOT=/Midgard/home/hongen/hybrid_teacher_cluster_runs
mkdir -p "$OUTPUT_ROOT"
```

Do not copy a second full nuScenes tree into the 100-GiB personal quota. Select
a permitted shared dataset location. Monitor checkpoint, log, cache, and output
usage under `/Midgard/home/hongen` throughout the experiment.

## 5. Verify CUDA through Slurm

No partition or account name is guessed. Check the current RPL documentation or
ask the administrator, then set the real values. `SLURM_ACCOUNT` and
`SLURM_CONSTRAINT` are optional; `SLURM_PARTITION` is intentionally mandatory.

```bash
export SLURM_PARTITION=<verified-partition>
# export SLURM_ACCOUNT=<verified-account>       # only if required
# export SLURM_CONSTRAINT=<verified-gpu-feature> # optional, e.g. after sinfo

export SLURM_CPUS=4
export SLURM_MEMORY=16G
export SLURM_TIME=00:20:00
export PRECISION=bf16
bash cluster_experiment/submit_slurm.sh verify
```

The verifier records the GPU model, memory, compute capability, BF16 support,
and a real CUDA tensor operation. RTX 2080 Ti does not provide native BF16, so
submit it with `PRECISION=fp16`. RTX 3080 may use `PRECISION=bf16` only when the
GPU verifier confirms support with the installed driver and PyTorch build.
Failure returns a non-zero job exit status.

The launcher always requests exactly one GPU, rejects requests above 48 hours,
and permits CPU, RAM, time, and constraint selection:

```bash
export SLURM_GPUS=1
export SLURM_CPUS=10
export SLURM_MEMORY=64G
export SLURM_TIME=24:00:00
```

## 6. Smoke test and production training

After the GPU verification succeeds, submit the real ten-sweep smoke protocol.
It runs four training and four validation samples through pretraining,
end-to-end finetuning, and the frozen 1x1 probe:

```bash
export PRECISION=fp16  # RTX 2080 Ti; use verified bf16 on a compatible RTX 3080
bash cluster_experiment/submit_slurm.sh smoke
```

The precision selection is passed to all three smoke stages. Do not start the
production run until each smoke summary says `"status": "complete"`.

Submit all production stages sequentially:

```bash
export SLURM_TIME=48:00:00
bash cluster_experiment/submit_slurm.sh all
```

Alternatively submit `pretrain`, then `finetune` and `linear-probe`, followed
by `evaluate`. Each stage resumes from `checkpoints/latest.pt` when present.
`SIGUSR1` and `SIGTERM` request a safe checkpoint after the current optimizer
update, which supports continuation across the 48-hour job limit.

Logs appear in:

- `$OUTPUT_ROOT/slurm_logs/` for Slurm stdout/stderr;
- each run's `console.log`, `status.json`, `metrics.jsonl`, TensorBoard events,
  epoch validation JSON, and `checkpoints/` directory;
- `$OUTPUT_ROOT/comparison/` for the final metric summary.

## Redeployment and recovery

After pulling changes, rerun the idempotent installer:

```bash
cd /Midgard/home/hongen/projects/DD2414-Project
git pull --ff-only
bash cluster_experiment/setup_rpl_env.sh
```

It does not delete or recreate a valid existing environment. If it rejects an
environment with the wrong Python version, create a differently named clean
environment or deliberately remove the old one after backing up anything
needed; the installer will not make that destructive choice automatically.

## Comparison boundary

The evaluator uses the same Vehicle BEV IoU formula and supervised loss
components as Simple-BEV. A numerical comparison to a published or checkpoint
baseline is defensible only when input resolution, split, schedule,
augmentation, and evaluation code also match. The generated
`comparison/official_metrics.md` states this boundary explicitly.
