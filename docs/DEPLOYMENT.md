# Deployment and publication

Copy `configs/deployment/example.json` outside Git and set this experiment's
absolute source/environment paths. Put actual data/checkpoint/output paths in
the unchanged branch CLI arguments, reviewed against its protocol. Empty launcher
means one interpreter; only the depth branch's existing distributed profile may
use its accepted torchrun arguments/rank counts. Do not retrofit DDP into other
experiments during cleanup. `tools/launch.py --config /private/config.json`
prints only; `--execute` is required to run. It never connects/submits jobs.

Original `cluster_rpl/` or `cluster_experiment/` controllers retain historical
user/node/branch guards and exact recovery scopes; these are labeled provenance,
not portable defaults. Use an admitted original source for existing jobs. Site
Slurm resources, partition, GPU permission, node-local data placement, driver and
mamba environment must match. No login-node computation/system installation.

Training queues no longer invoke publishers automatically. Legacy generated
artifact publication is retired. `tools/publish_results.py`
accepts only the reviewed benchmark summary and handoff. Default prints commands;
`--execute` is required to publish, without force-add or main writes. Historical
publisher entries now raise a retirement message; use the tag for old behavior.
The cleanup did not sync or change remote jobs.
