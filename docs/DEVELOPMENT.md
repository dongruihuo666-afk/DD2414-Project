# Installation, checks and maintenance

The existing workstation uses Python 3.10.14, torch 2.5.1 (CUDA 12.4 build),
torchvision 0.20.1, NumPy 1.23.5, Pillow 9.4.0, nuScenes SDK 1.1.11. These were
not upgraded. Use separate environments per experiment, or explicitly reinstall
the matching editable checkout. A wheel imports tensor helpers independently;
training needs the matching source checkout for scripts/config/assets and checks
actual package bytes through the existing implementation-hash config field.

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dd2414-wheel .
dd2414 --help
python tools/testing/run_cpu.py
python tools/quality/check_source.py
```

`pyproject.toml` separates core, train, test and site-tool scopes. The legacy
`requirements.txt` is retained for original/LYFT provenance, not a recommended
environment lock; do not install its Git Lyft SDK over the nuScenes environment.
Existing branch-specific mamba YAML (where present) remains under its original
deployment directory. No system drivers/modules or external repository are
installed by package import/help. Site Slurm/SSH/rsync are not pip dependencies.

The explicit CPU runner chooses existing unittest modules, excluding real-data
and plotting programs. Local Gloo and worker-prefetch tests need socket/shared
memory permissions. They do not need a GPU, weights, full dataset or SSH.
`tests/unit` adds boundary/provenance checks. Numerical parity uses fixed CPU
inputs against THIS branch's preserved source, not another experiment's code:

```bash
python tests/parity/probe.py --implementation legacy --source-root /path/to/original > /tmp/before.json
python tests/parity/probe.py --implementation package --source-root "$PWD" > /tmp/after.json
python -c 'import json; assert json.load(open("/tmp/before.json")) == json.load(open("/tmp/after.json"))'
```

`scripts/test_bevcar_encoder_smoke.py`, `test_bevcar_voxel_adapter.py`,
`test_radar_point_encoder.py`, `test_radar_velocity_rotation.py` and
`smoke_test_mini.py` are opt-in integration/experiment tools. They can need
data/CUDA or generate plots and are not blanket unit discovery. Their numerical
coordinate/state regression counterparts remain in the explicit CPU suite.
The old rotation entry wraps `tools/visualization/radar_velocity_rotation.py`.
Existing plotting/report commands are opt-in and use ignored external outputs.

New formal modules must not depend on experiment/demo/tiny/smoke entrypoints or
training drivers. Legacy helper aliases preserve import/monkeypatch compatibility.
Keep official `nets`, `utils`, dataset, state names, manifest order/hash and
source/admission checks stable. No baseline scores may be attributed to a
single research factor when initialization/resolution/readout/budget differ.

CI covers pinned CPU installation, imports/help, source boundaries and CPU
regressions. A local pass is not a claim that hosted CI, CUDA or full-data
inference has passed. No expensive full retraining is needed for code layout.

## Recorded cleanup validation

This branch's final CPU suite ran 192 tests;
all applicable checks passed. Original-versus-extracted fixed-input output and
gradient/state hashes match exactly in 10 fixture groups. All 38 pinned
official/model/data/manifest files retain their original bytes.
Editable/wheel build/install/import/help and package/source content matching
passed in a temporary branch-specific environment. Real official RGB checkpoint
strict loading passed for 677 full-model state items and 568 encoder tensors.
Syntax, consolidated-document links and scoped Git whitespace checks passed.
No full-data/CUDA training, remote job operation or hosted CI result is claimed.
