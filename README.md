# DD2414: Camera and Radar BEV Learning

**Experiment branch:** `dd2414-mini-baseline`

This branch contains the radar-scaling experiments and the completed ten-sweep full-data study initialized from the official RGB encoder.
Each experiment has its own branch, configuration and results.
`main` contains the original Simple-BEV reference.

## Getting started

Use Python 3.10, PyTorch 2.5.1 and torchvision 0.20.1. In an environment with
the dependencies already installed:

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 train --help
python tools/testing/run_cpu.py
```

See [development](docs/DEVELOPMENT.md) for dependencies and checks, and
[the experiment](docs/EXPERIMENT.md) for initialization, training and evaluation.
Training requires local nuScenes data, teacher weights and external sources.

## Layout

| Path | Contents |
| --- | --- |
| `src/dd2414/` | Model wrappers, teachers, losses, data interfaces and training utilities |
| `nets/`, `utils/`, `nuscenesdataset.py` | Simple-BEV implementation and dataset loader |
| `configs/` | Data manifests, experiment settings and deployment examples |
| `scripts/` | Training, evaluation and historical experiment commands |
| `tests/`, `tools/` | Regression tests, checks and command-line tools |
| `docs/` | Experiment, interface, setup and recovery documentation |
| `results/` | Selected benchmark summaries |

[Branches](docs/BRANCHES.md) describes the experiment variants.
[Interfaces](docs/INTERFACES.md) documents input fields, coordinates and outputs.
[Deployment](docs/DEPLOYMENT.md) covers launching jobs and publishing summaries.
[Reproducibility](docs/REPRODUCIBILITY.md) records source snapshots and asset hashes.
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) records the latest engineering checks.

Generated artifacts, logs, weights, personal settings and old meeting records
are ignored. Original records remain available in the source snapshots.

## Attribution

Based on [aharley/simple_bev](https://github.com/aharley/simple_bev), commit
`be46f0ef71960c233341852f3d9bc3677558ab6d`:
**Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?**,
Adam W. Harley, Zhaoyuan Fang, Jie Li, Rares Ambrus and Katerina Fragkiadaki,
arXiv:2206.07959. See [LICENSE](LICENSE). BEVCar, DINOv2 and Depth Anything retain
their respective licenses.
