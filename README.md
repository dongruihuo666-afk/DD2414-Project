# DD2414: camera/radar BEV research

This course research project preserves the original Simple-BEV implementation
and adds opt-in camera/radar representation learning: frozen DINOv2 teachers,
image and radar-supported fused-BEV distillation, BEVCar VoxelNet, temporal
radar sweeps, depth-aware lifting and supervised downstream readouts.

Use **`codex/team-integration-20261113`** to review the validated integration layout.
`main` retains the upstream Simple-BEV history; it has not been merged with this
cleanup. The complete pre-cleanup source is tagged
`pre-integration-refactor-20261113T144801Z`; the depth-source variant has the
`-depth-study` suffix. [Backup refs and exclusions](docs/integration/BACKUP.md).

## Start here

In an existing compatible Python 3.10 / Torch 2.5.1 / torchvision 0.20.1
environment (do not replace the environment of a running experiment):

```bash
python -m pip install --no-deps --no-build-isolation -e .
dd2414 train --help
dd2414 evaluate --help
python tools/quality/check_source.py
python tools/testing/run_cpu_regressions.py
```

Dependency preparation, wheel use and full-data command examples are in
[installation](docs/INSTALL.md). Plain import and CLI help never download
weights, start training or contact a cluster. Actual training requires explicit
data/source/output paths and assets. The legacy `requirements.txt` is retained
for provenance; it is not the recommended DD2414 environment lock.

## Integration surface

| Location | Responsibility |
| --- | --- |
| `src/dd2414/` | Named batch interface, teacher/target tools, losses, model wrappers, objectives, evaluation and state runtime |
| `nets/`, `utils/`, `nuscenesdataset.py` | Preserved Simple-BEV paths and checkpoint keys; installed legacy namespaces |
| `configs/` | Frozen data manifest/protocol plus separate experiment and deployment templates |
| `tests/`, `tools/testing/` | Unit tests, explicit CPU regressions, numerical parity |
| `scripts/` | Compatible research/training entrypoints; common implementations imported from the package |
| `tools/` | Quality checks, explicit Git publication, portable launch planning and visualization |
| `cluster_rpl/` | Historical site-specific controllers/recovery tools; opt-in, scope documented |
| `experiments/`, `research/`, `artifacts/` | Research index, provenance archives and tracked selected results/reports |
| `docs/` | Current interfaces, architecture, setup, protocols and indexed history |

The other team's primary contract is [docs/INTERFACES.md](docs/INTERFACES.md):
field names, shapes, dtype/device policies, frames, metres/seconds, camera order,
BEV geometry and feature/logit distinctions. Example feature extraction is
documented there. There is no HTTP service or newly invented task interface.

## Research status and reproducibility

The committed official full-data study record contains all eleven stage results;
the 50-epoch random-initialized metric-depth study is a separate protocol.
[Current protocol index](docs/README.md) distinguishes them from historical
mini/subset/low-resolution experiments. Frozen probe IoU, fine-tuned segmentation,
GT-box velocity EPE and radar-target loss differences answer different questions.
Existing records are snapshots, not live process-status claims.

No model math, loss weight, manifest order/hash, training budget or official
checkpoint key was intentionally changed. Fixed-input CPU parity is exact;
CUDA/full-data revalidation is not claimed. Original-source exact resume uses
the original tag or immutable execution directory, preserving optimizer,
sampler, RNG, counters and strict source checks. See
[compatibility](docs/integration/COMPATIBILITY.md) and
[the current handoff](PROJECT_HANDOFF.md). Ordinary training does not publish
Git changes; publication is an explicit separate command.

## Upstream attribution

Derived from [aharley/simple_bev](https://github.com/aharley/simple_bev), upstream
commit `be46f0ef71960c233341852f3d9bc3677558ab6d`, **Simple-BEV: What Really
Matters for Multi-Sensor BEV Perception** by Adam W. Harley, Zhaoyuan Fang,
Jie Li, Rares Ambrus and Katerina Fragkiadaki. The original [LICENSE](LICENSE), source headers
and [original README/citation](docs/archive/pre-integration/README.md) are
retained. [External assets](docs/EXTERNAL_ASSETS.md) records BEVCar, DINOv2 and
Depth Anything provenance; their own licenses apply.
