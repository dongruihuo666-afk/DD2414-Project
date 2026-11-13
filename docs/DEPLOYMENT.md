# Deployment, execution and publication boundaries

The portable templates contain no username, fixed compute node, SSH identity or
personal storage path. Copy `configs/deployment/portable.example.json` outside
Git and replace its absolute paths. Keep dataset/manifest information separate
from model arguments (`configs/experiments/*.json`). The JSON examples reproduce
the existing random 50-epoch profile, not an authorization to launch more runs.

```bash
python tools/launch_training.py \
  --deployment /private/deployment.json \
  --experiment configs/experiments/random-metric-bev-dino-only.json
```

This only prints an argv-quoted plan. `--execute` explicitly runs a NEW experiment
in that configuration's environment; the trainer still performs full profile,
source/hash/weight/manifest admission. Default planning does not inspect a GPU,
connect to a cluster, create a run directory or change Git. Existing outputs
are rejected for new runs; exact new-source restart requires `--resume` and its
checkpoint. Never use these examples to migrate an ongoing old-source run.

`configs/deployment/slurm.example.sh` is a **template, not a submitted job**.
Select site-specific partition, allocation limits, time and CPU/RAM resources;
match allocated GPUs to `local_ranks`. Create the `logs/` directory before
submission. With node-local data, pin the allocation to the node containing the
verified personal data copy. Mamba environments/CUDA must match that node's
driver; do not install system drivers or run compute on a login node. This
cleanup neither logs in to the cluster nor changes its existing allocations.

## Historical site controllers

`cluster_rpl/` retains original examples, source-pinned execution roots,
Slurm/account ownership checks, admission reports and narrowly scoped recovery
helpers. Their personal paths and fixed branch guards are historical deployment
inputs, not defaults for another team. `continue_recovered_depth.py` is only
for its admitted partial parents with recorded failure `FAILED|75:0`; it must
not retry arbitrary/OOM/cancelled failures or grant extra epochs. The retired
three-way warm-start plan is in the archive; the current two-arm random-depth
protocol remains explicit. Do not rewrite old paths/hashes to bypass recovery.

The only centralized `sys.path` support needed by new source-checkout research
wrappers is `scripts/_bootstrap.py`. Legacy commands keep historical support;
an explicit external teacher loader may still add its pinned source directory.

## Explicit result publication

Ordinary training and queue execution never stage/commit/push Git changes.
Historical local queue controllers now only record a manual publication command.
For an established historical study, review its existing publisher's file scope
and branch guards, then print the command:

```bash
python tools/publish_results.py --study official --study-dir /outputs/study
```

Only adding `--execute` invokes that Git publisher. It remains subject to its
original guards and may reject this integration branch; do not bypass them.
For new studies, separately review and commit selected metrics/provenance on
the appropriate feature branch. Reports/logs/checkpoints are independent of a
successful push. Publication, deployment and model execution are different tasks.
