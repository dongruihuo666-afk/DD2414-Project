# Deployment and result publication

## Launching an experiment

Copy `configs/deployment/example.json` to a local settings file. Set the Python
interpreter, source checkout, branch, command and arguments for your experiment.
Print the command first:

```bash
python tools/launch.py --config /path/to/deployment.json
```

Add `--execute` to run it. The tool checks the branch before launching and uses
the selected checkout as its working directory. An empty `launcher` starts one
Python process. The depth branch also supports its documented torchrun profiles.

For Slurm, request resources under the site's policy and run on a compute node.
With node-local data, select the node containing the verified copy. Match the
mamba environment and CUDA build to the node's driver. Site-specific scripts
remain in `cluster_rpl/` or `cluster_experiment/`; their fixed paths, admission
checks and recovery rules apply to the original runs they describe.

## Publishing a summary

Training writes results locally. To publish a reviewed summary, first inspect
the planned Git commands:

```bash
python tools/publish_results.py --path results/benchmarks.json
```

The tool accepts `results/benchmarks.json` and `PROJECT_HANDOFF.md`. Add
`--execute` to commit and push on the selected experiment branch. Generated
artifacts remain ignored. Older `scripts/publish_*.py` entrypoints are retired;
their original implementations are available in the source snapshots.
