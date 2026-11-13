# Development

## Environment and installation

The workstation uses Python 3.10.14, PyTorch 2.5.1 with CUDA 12.4,
torchvision 0.20.1, NumPy 1.23.5, Pillow 9.4.0 and nuScenes SDK 1.1.11.
Use the existing matched environment for a checkpoint continuation. Cluster
mamba profiles, where available, are in the branch's deployment directory.

`pyproject.toml` groups core, training and test dependencies. The upstream
`requirements.txt` also contains the Lyft SDK and is retained for provenance;
use a separate environment for that path.

With dependencies prepared:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dd2414-wheel .
dd2414 --help
```

Training from a wheel requires a matching source checkout for scripts and
configuration. Set `DD2414_PROJECT_ROOT` when it cannot be located automatically.
Help commands parse arguments before importing the training dependencies.

## Checks

```bash
python tools/testing/run_cpu.py
python tools/quality/check_source.py
python tools/quality/check_links.py
```

The CPU runner includes unit, checkpoint, coordinate and protocol checks. Local
Gloo and worker-prefetch tests require socket/shared-memory access. They use
CPU fixtures and mocks rather than a full dataset, downloaded weights or SSH.
The GitHub workflow runs the same checks on CPU.

The data/CUDA programs `scripts/test_bevcar_encoder_smoke.py`,
`test_bevcar_voxel_adapter.py`, `test_radar_point_encoder.py`,
`test_radar_velocity_rotation.py` and `smoke_test_mini.py` are run explicitly.
The rotation program is implemented in `tools/visualization/` with its original
script retained as a wrapper.

For numerical comparison with an original source checkout:

```bash
python tests/parity/probe.py --implementation legacy --source-root /path/to/original > /tmp/before.json
python tests/parity/probe.py --implementation package --source-root "$PWD" > /tmp/after.json
python -c 'import json; assert json.load(open("/tmp/before.json")) == json.load(open("/tmp/after.json"))'
```

These compare fixed CPU outputs, gradients and restored state. Full-data CUDA
inference is a separate check. See the handoff for the checks actually run.

## Code organization

Put reusable implementations in `src/dd2414` and keep experiment commands in
`scripts/`. Existing aliases support old imports and test patches. Preserve
model state names, source checks, coordinate conventions and manifest ordering.
Changes to mathematical helpers need numerical and gradient comparisons;
changes to restore logic need optimizer, sampler and RNG regression checks.
