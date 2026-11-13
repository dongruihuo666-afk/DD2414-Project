# Installation and execution

The verified workstation environment is Python 3.10.14, PyTorch 2.5.1,
torchvision 0.20.1, NumPy 1.23.5 and Pillow 9.4.0. This cleanup does not upgrade
Torch, CUDA, drivers or the environment used by existing runs. The CPU-only CI
wheel and the local CUDA wheel have different capabilities despite the same
public Torch version. Choose the appropriate build explicitly when preparing
a new machine; do not replace a working cluster environment automatically.

## Install in an existing compatible environment

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 --help
dd2414 train --help
dd2414 evaluate --help
```

Use `--no-deps` only after preparing dependencies. `pyproject.toml` separates
core tensor/runtime packages, `[train]` (SDK, logging, teachers/official model
dependencies) and `[test]`. `[cluster]` intentionally has no Python packages:
Slurm, SSH and rsync are provided by the site. `requirements.txt` remains the
legacy upstream/LYFT list, including its unpinned Git SDK; it is **not** the
recommended reproducible DD2414 environment. Do not install the Lyft SDK over
the nuScenes environment. Upstream deformable-attention CUDA extensions under
`nets/ops` are optional for other architectures and are not built on import.

For an isolated editable check without modifying the workstation environment:

```bash
/path/to/bev/bin/python -m venv --system-site-packages /tmp/dd2414-dev
/tmp/dd2414-dev/bin/python -m pip install --no-deps --no-build-isolation -e .
```

## New training and downstream readouts

A mamba profile with previously admitted CUDA 12.1 cluster pins is available in
[configs/environments](../configs/environments/README.md). It is separate from
the workstation's working CUDA 12.4 build and was not installed over it.

`dd2414 train` dispatches the existing `train_multisweep_teacher.py` driver;
flags/defaults are identical and now parsed in a lightweight module. It requires
a source checkout for manifests, historical drivers and provenance. An installed
wheel may use `DD2414_PROJECT_ROOT=/path/to/matching/checkout`; code bytes must
match. Run from that checkout so unchanged historical relative paths resolve.
The Python feature API and saved-array evaluation also work outside a checkout.

Example NEW official-initialized full-data run (not an automatic experiment or
a resume of any historical run):

```bash
dd2414 train --mode pretrain --data-root /data/nuscenes \
  --bevcar-source /sources/BEVCar \
  --manifest configs/radar_scaling_manifest_seed125.json \
  --run-dir /outputs/new-double-distill \
  --official-camera-checkpoint /weights/simplebev-rgb.pth \
  --image-resolution 448 800 --teacher-resolution 448 798 \
  --nsweeps 10 --velocity-mode full --depth-mode none \
  --epochs 5 --batch-size 1 --num-workers 0
```

The `random-metric-bev` profile is a separate fixed 50-epoch protocol with
2/5/10 local DDP ranks, BF16 and per-rank batch one. Its unchanged admission
checks reject unsupported budgets/initialization/weights. Review
[deployment](DEPLOYMENT.md) and the current protocol before launching it.
`--distributed` dispatches the DDP driver under `torchrun`; a plain help command
does not allocate GPUs, load teachers, download weights or contact a cluster.

Downstream modes are `frozen-probe`, `unfrozen-probe`, `finetune` and
`object-motion`; each requires a compatible **completed** parent checkpoint.
Specify all parent profile settings unchanged. `--resume` is exact state resume,
not a way to initialize a new phase. Old experiments must use their original
tagged/fixed source; [compatibility](integration/COMPATIBILITY.md) explains why.

## Evaluation and feature extraction

```bash
dd2414 evaluate --arrays /outputs/prediction_target_valid.npz
```

The NPZ contains equal-shaped `logits`, `target`, `valid` arrays `[N,1,Z,X]`.
This computes official binary counts/IoU from saved outputs; it does not load a
model or claim an independent dataset evaluation. For actual model features,
use the [Python interface](INTERFACES.md). The official Simple-BEV checkpoint
path stays `eval_nuscenes.py` and the original download/eval scripts; their
weights, resize convention and init directory must match the benchmark record.
`eval.sh` is an upstream convenience profile, not a universal reproduction.

## Build artifact

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dd2414-wheel .
```

Only the formal package, explicit legacy model/util namespaces and dataset/state
modules enter the wheel. Data, weights, history, credentials, report files and
Slurm scripts do not. A wheel alone cannot resume a historical source-pinned run.
