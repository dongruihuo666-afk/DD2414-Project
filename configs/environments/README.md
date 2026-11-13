# Environment profiles

The workstation environment already works with Python 3.10.14, torch 2.5.1
(CUDA 12.4 build), torchvision 0.20.1 and the versions recorded in installation
docs. It was neither recreated nor upgraded for cleanup. Tests/builds use
temporary venvs inheriting its packages.

`rpl-cu121.yml` copies the previously admitted cluster mamba package pins
(CUDA 12.1 wheels), with only an isolated new environment name. It is a profile
for a **new** compatible cluster environment, not a command to replace a running
environment or a claim that a new solve was executed during cleanup:

```bash
mamba env create -f configs/environments/rpl-cu121.yml
mamba activate dd2414-integration
python -m pip install --no-deps --no-build-isolation -e .
```

Use the existing matched environment for a historical resume. Check node driver
compatibility and site module policy before creating a new GPU environment.
Slurm/SSH/rsync are site tools, not pip packages. The original
`cluster_rpl/environment.yml` and legacy `requirements.txt` remain for their
historical scopes; no CUDA/system dependency is upgraded by an ordinary import.

CPU CI explicitly installs torch/torchvision CPU wheels at the same public
versions, then package train/test dependencies. This validates engineering
interfaces and numerical CPU fixtures; it does not certify CUDA parity.
