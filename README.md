# DD2414 — independent camera/radar BEV experiment

**Branch: `dd2414-mini-baseline`**

Workstation experiment history, full-data radar scaling and the completed official-RGB-initialized ten-sweep study.

Each branch is an independent experiment, not a staging area for one merged
model. `main` stays the original Simple-BEV reference. This cleanup creates no
new experiment branch and does not merge histories or import another branch's
newer model, data ordering, initialization, objective or training budget.

## Install and run

Use an existing compatible Python 3.10 / Torch 2.5.1 / torchvision 0.20.1
environment; do not change the environment or fixed source of a running job.

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 --help
dd2414 train --help
python tools/testing/run_cpu.py
python tools/quality/check_source.py
```

`--no-deps` assumes dependencies are already prepared. See
[installation/testing](docs/DEVELOPMENT.md). Import/help does not download weights,
start training or contact a cluster. Actual runs need explicit data, source,
checkpoint and output paths. Read the unchanged branch-specific flags/protocol.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/dd2414/` | This branch's common teachers/targets, losses, wrappers, runtime and named batch interface |
| `nets/`, `utils/`, `nuscenesdataset.py` | Preserved official path and branch-local research implementation |
| `configs/` | Unchanged data manifest/protocol; separate optional deployment paths |
| `scripts/` | Compatible training and opt-in historical experiment commands |
| `tests/`, `tools/` | CPU/state/parity checks, explicit deployment/publication and report tools |
| `docs/` | Consolidated project, experiment, interface and recovery documentation |
| `results/` | Only a small curated benchmark summary, when this branch has one |

Start with [the branch index](docs/BRANCHES.md), [current experiment](docs/EXPERIMENT.md),
[tensor/API contract](docs/INTERFACES.md), [engineering checks](docs/DEVELOPMENT.md)
and [recovery/artifact policy](docs/REPRODUCIBILITY.md). The short
[handoff](PROJECT_HANDOFF.md) records actual checks and limitations.

## Records and generated files

Past day plans, per-action journals, meeting PDFs, bulk images, raw JSON/JSONL
and runtime output no longer belong to this branch's active Git tree. They
remain untouched locally and recoverable from the verified pre-cleanup tag.
`artifacts/` is ignored without whitelist exceptions. Necessary configuration
is in `configs/`; this is separate from generated reports and model parameters.
No data, weights, cache, environment or cluster job was deleted or stopped.

## Attribution

Derived from [aharley/simple_bev](https://github.com/aharley/simple_bev),
`be46f0ef71960c233341852f3d9bc3677558ab6d`. **Simple-BEV: What Really Matters
for Multi-Sensor BEV Perception?**, Adam W. Harley, Zhaoyuan Fang, Jie Li,
Rares Ambrus and Katerina Fragkiadaki, arXiv:2206.07959. The original
[LICENSE](LICENSE) and source headers are retained. External BEVCar, DINOv2 and
Depth Anything licenses apply to their respective sources/assets.
