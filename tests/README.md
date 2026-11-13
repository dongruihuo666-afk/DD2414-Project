# Test and experiment boundaries

| Category | Entry | Requirements / effects |
| --- | --- | --- |
| Fast boundary/unit tests | `python -m unittest discover -s tests/unit -v` | Installed package and CPU Torch; temporary files only |
| Preserved numerical/state regressions | `python tools/testing/run_cpu_regressions.py` | CPU train dependencies, local sockets/shared memory for two-process Gloo and DataLoader; mocked SDK/jobs, no SSH |
| Before/after parity | `tests/parity/probe.py` | Preserved source checkout plus matching CPU dependencies; outputs JSON hashes |
| Real-data/CUDA integration | listed in `experiments/README.md` | Explicit assets/data/output arguments; **not** ordinary test discovery |
| Visualization/report tools | `tools/visualization`, historical `scripts/render_*`, `scripts/build_*` | Explicit results/outputs; no scientific pass claim |
| Static checks | `python tools/quality/check_source.py` | Standard library only; syntax and package import boundary |

`pytest` defaults to `tests/unit` deliberately. Historical unittest filenames
under `scripts/` and `cluster_rpl/` stay import-compatible; the curated CPU
runner names them explicitly. It includes strict checkpoint/source admission,
sampler/optimizer/RNG resume, actual two-process CPU gradient synchronization,
radar/velocity transforms, depth probability/projection/gradients and manifest
checks. Cluster helpers are tested with mocks; no Slurm job is submitted.

Files named `test_bevcar_encoder_smoke.py`, `test_bevcar_voxel_adapter.py`,
`test_radar_point_encoder.py`, `test_radar_velocity_rotation.py` and
`smoke_test_mini.py` are historical **integration/experiment programs**, not
safe automatic unit discovery. Their outputs can include plots or training
steps. Names alone do not determine classification. The numeric rotation/depth
regressions in the CPU suite are kept separately from those programs.

CPU CI installs pinned runtime/train dependencies and runs these checks without
nuScenes, downloaded model weights, a GPU or cluster credentials. A successful
local run does not claim a hosted CI run, full-data inference or CUDA/DDP parity.
Exact historical restore still requires its original source (see compatibility).
