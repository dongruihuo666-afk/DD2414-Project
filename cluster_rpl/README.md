# RPL full-data double-distillation deployment

Integration note: this directory preserves site- and source-specific controllers,
including historical personal paths, branch guards and recovery policies. Do
not treat these as portable defaults or rerun recorded jobs. See
[portable deployment](../docs/DEPLOYMENT.md). The cleanup did not synchronize,
submit, cancel, resume or modify any remote job or immutable execution root.

For the current **random ResNet101 / 50-epoch / two-treatment metric BEV depth**
study use [CLUSTER_RANDOM_DEPTH_50EPOCH_PLAN.md](../CLUSTER_RANDOM_DEPTH_50EPOCH_PLAN.md),
`random_depth_study.example.json`, `audit_random_depth.py` and
`random_depth_job.sh`. The older deployment/three-way continuation instructions
below are historical alternatives, not the new experiment's initialization.

On the isolated `experiment/rpl-depth-10sweep-velocity-study` branch, this guide
describes the inherited single-GPU deployment, not a ready automatic depth
queue. The branch now adds opt-in two-GPU DDP PRETRAINING, a three-treatment
preview and a signal-safe wrapper; downstream readouts remain single-GPU.
See `CLUSTER_DEPTH_RUN_PLAN.md` for exact commands, changed global batch,
CPU-only validation and unfinished CUDA/deployment gates. After the owner's
subsequent authorization, the cluster checkout was deployed, full data/asset
audit passed, and all three bounded two-GPU update-two/new-process-update-three
pilots passed. See `artifacts/rpl_depth_admission_20261106.json` and the depth
plan's latest runtime checkpoint. This is not completed five-epoch training or
a semantic result. The formal applications and their recorded hold state are
listed in that plan and `PROJECT_HANDOFF.md`.

Branch: `experiment/rpl-full-data-double-distill`.

This opt-in deployment leaves the workstation's `dd2414-mini-baseline` checkout,
systemd services, protocol and runs untouched. It does not restore the retired
`experiment/official-rgb-double-distill-cluster` branch. The cluster checkout
is `/Midgard/home/hongen/projects/DD2414-Project` (singular `Project`).

## Execution scope and resource profiles

On 2026-11-05 the owner replaced per-command prompts with standing approval
for routine inspection, project synchronization, personal-directory environment
and data preparation, and the approved single-GPU workflow. **Ask before high-
privilege operations, interference with existing jobs, other people's files,
shared configuration or broader destructive operations.** The separate request
to delete only `dd2414-hybrid` does not authorize deleting other environments.
Keep commands/job logs and report failures. Routine commands no longer need
individual prompts under that working agreement.

No script here connects through SSH, pushes to GitHub, automatically submits a
queue, cancels jobs, requeues jobs or chooses a node. Each real submission needs
an explicit task and `--execute`. Default `plan` and `submit` only print.
GPU counts describe implementation profiles, not a repository-wide single-GPU
safety cap: inherited `control.py` requests one GPU; depth continuation uses
two DDP processes/GPUs; downstream readouts use one GPU. GPU memory is not
pooled. The repository's `AGENTS.md` and Stop hook do not impose a GPU-count
limit. Site rules and platform/tool permissions are external to these files;
editing this guide does not change them. Current application/release status
and historical execution records have one entry point: `PROJECT_HANDOFF.md`.

Depth-study preview (no torch import, cluster access or allocation):

```bash
python cluster_rpl/depth_study.py --config cluster_rpl/depth_study.example.json
```

Do not use the inherited `control.py` task list or its old admission report as
evidence that the depth DDP modes have passed full-model admission. The new
matrix has three two-GPU continuations and twelve single-GPU readouts; it is
not an automatic queue. Follow the depth plan's deployment gates first.

## Scientific scope

All formal runs use the official 700-scene train / 150-scene validation split,
28,130 / 6,019 samples, seed 125, 448x800 camera input, 448x798 DINO input,
ten radar sweeps, no depth module, and batch size one. No 64/4096 subsets exist
in this deployment. These are validation results, not a new blind test.

The camera encoder is initialized strictly from the nuScenes-trained official
Simple-BEV RGB checkpoint. Only its encoder tensors are loaded; radar/fusion
and other student components are not initialized from the full official model.
The frozen DINOv2 ViT-S/14 teacher supervises the trainable camera encoder and
radar-supported fused BEV. The image/BEV/motion-auxiliary weights stay
0.2/0.8/0.5. The two independently trained treatments provide radar velocity or
hide its input channels throughout training and evaluation.

Budgets: pretraining five epochs; each frozen 1x1 probe ten epochs; end-to-end
vehicle segmentation and GT-box object-velocity readout five epochs each.
Vehicle IoU follows Simple-BEV global valid-cell counts and sigmoid rounding;
precision, recall and F1 are secondary. Object EPE uses provided GT boxes, not
object detection/NDS. Radar-derived auxiliary losses are not semantic accuracy.

Priority is dependency-aware reverse order: hidden-velocity pretraining,
its segmentation, object motion, fused probe, camera probe; then missing
velocity-present tasks in that order; initial encoder probe last if missing.
Refresh completion inventory before each submission to avoid duplicating
workstation jobs. Never transfer a checkpoint that is actively being written.
The inventory records the active workstation task as a snapshot, and submission
refuses that task until refreshed. It is not a distributed reservation/lock:
the unchanged workstation queue may later reach an unstarted cluster task.
Before a long formal launch, explicitly coordinate task ownership with the
owner; never silently stop/reorder the workstation queue. Do not claim that a
snapshot alone guarantees no future cross-host overlap.

## 1. Inspect and synchronize the existing clone

Inspect the clone and account jobs before switching:

```bash
git -C /Midgard/home/hongen/projects/DD2414-Project status --short --branch
```

```bash
squeue -u hongen
```

Confirm that no task is reading the checkout before switching it. If it is in
use, create a separate worktree after owner approval instead of altering it.
Do not reset, overwrite, clean or automatically stash teammate changes.
After the new branch has been pushed from the workstation:

```bash
git -C /Midgard/home/hongen/projects/DD2414-Project fetch origin
```

For an old single-branch clone, first inspect `git config --get-all
remote.origin.fetch`. If it only maps the historical branch, add a mapping
for `refs/heads/experiment/rpl-full-data-double-distill` to
`refs/remotes/origin/experiment/rpl-full-data-double-distill` with
`git config --add remote.origin.fetch SOURCE:DESTINATION`, preserving the old
rule, and fetch again. A manually fetched remote ref alone may not be recognized
for `--track` without this mapping. Do not add duplicate mappings blindly.

If the branch is absent locally and the worktree is clean (moria's older Git
does not implement `switch`):

```bash
git -C /Midgard/home/hongen/projects/DD2414-Project checkout -b experiment/rpl-full-data-double-distill --track origin/experiment/rpl-full-data-double-distill
```

```bash
git -C /Midgard/home/hongen/projects/DD2414-Project rev-parse HEAD
```

Compare to the verified pushed commit. A future update on the same branch uses
`git pull --ff-only`, only when no running job uses the checkout. Do not use
the workstation's `scripts/start_official_full_data.sh` on RPL: it is a systemd
launcher and its queue waits for a 22,000 MiB physical-GPU-0 availability check.

## 2. Choose a node and inspect storage

Check `rplgpu`, `rplquota`, partition/account settings and the proposed node.
The supplied manuals are not live hardware inventories.
Prioritize a compatible RTX 3080 node, but measure actual VRAM and free disk.
An idle node does not guarantee immediate scheduling. Turing RTX 2080 Ti does
not have native BF16; never infer precision support from driver version alone.

`/Midgard/home/hongen` has a 100 GiB quota, so do not stage the full dataset
there. Full nuScenes size must be remeasured on the workstation before selecting
a node-local target, with room for outputs and checkpoints. Local disks are not
shared: all jobs must use the configured `--nodelist` data node.

Example CPU-only preparation allocation (replace the node after inspection;
obtain approval before running it):

```bash
salloc --nodes=1 --nodelist=SELECTED_NODE --gres=gpu:0 --cpus-per-task=4 --mem=32G --time=02:00:00
```

An allocation does not universally move the current shell to a compute node.
Use an approved `srun` command inside it and check `hostname`. Do not hold an
idle GPU while installing packages/copying data. Interactive allocations have
the site's six-hour limit; release them when done. Longer transfers need
approved batching/CPU batch jobs, not forgotten tmux sessions.

Suggested layout on the chosen node:

```text
/local_storage/users/hongen/
  datasets/nuscenes/{samples,sweeps,maps,v1.0-trainval}/
  dd2414_assets/{official_rgb.pth,BEVCar/}
  dd2414_runs/official_rgb_double_distillation_10sweep/
```

## 3. Upload data and non-Git dependencies

Before copying, confirm that an allocated compute node is reachable by the
site-approved transfer route. The login key lives on the workstation under
`/home/students2026/Hongen_sshkey/.ssh/`; the older config's `~/.ssh/...`
IdentityFile does not identify that file. Do not copy private keys into the
project/cluster, forward an agent unnecessarily, disable host-key checks, or
assume direct compute-node SSH is permitted. The route and chosen hostname
are prerequisites to an exact rsync command. One option is an rsync server
invoked by CPU-only `srun` through the trusted moria session: no compute-node
key forwarding or idle placeholder allocation is needed. Test a small transfer
first, and only document the route as working after that real test.
Use `srun --unbuffered` for this binary protocol: the default buffered mode
stalled the initial bounded dry run, whereas the unbuffered dry run succeeded.
Each transfer must request explicit CPU, RAM, node and <=48-hour time limits;
no GPU or idle allocation holder is needed. A long-running workstation rsync
must be persisted by a service, not depend on an interactive agent tool call.

Use a dry-run inventory, then a resumable rsync copy
with `--partial --info=progress2`, never `--delete`. Preserve complete source
layout, particularly historical sweeps for the ten-sweep input. Do not delete
the workstation source. A size/mtime comparison alone is not byte verification:
use transfer checksums for the payload and a final checksum audit where feasible.
Millions of files can make metadata overhead dominant; if needed, request a
separate approved chunk/archive plan with checksums and enough staging space.

Also copy these separately; Git does not contain them:

* Official RGB checkpoint to configured `camera_checkpoint`; SHA-256 is pinned
  in `configs/full_data_official_protocol.json`.
* Official BEVCar source, including provenance, at commit
  `29cacda3bc5416d47428c1d0f017527acad34f90`; only `nets/voxelnet.py` is imported.
* The workstation's exact DINO hub source to
  `PROJECT/.cache/torch/hub/facebookresearch_dinov2_main/` and weight to
  `PROJECT/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`.
  `assets.json` pins the actual Python-tree digest and weight hash.

Those DINO assets are small enough for the shared project cache. Do not copy
the full teacher-feature cache, environment directory, logs, credentials or
the entire workstation checkout.

## 4. Build the isolated mamba environment

Mamba 2.1.1 was observed on the account. Reuse Miniforge at
`/Midgard/home/hongen/miniforge3`. Do not reinstall it or mutate base.
`environment.yml` pins Python 3.10.14, torch 2.5.1, torchvision 0.20.1,
NumPy 1.23.5, Pillow 9.4.0 and nuScenes devkit 1.1.11. It uses the official
CUDA 12.1 torch build for the cluster, not the workstation CUDA 12.4 build;
record this numerical-environment difference. Initial CPU setup job 1250185
failed solving conda dependencies: old Pillow required an old libxcb that
conflicted with libgl, and scipy 1.15.3 was unavailable in that resolution.
The revised YAML uses mamba for Python/user libraries and pinned PyPI/official
CUDA wheels for the numerical packages. Setup job `1250190` subsequently
completed in 8m48s. CPU validation job `1250197` passed `pip check`, imports,
a frozen DINO forward and all 66 regression tests. PyYAML is pinned explicitly
because the environment contract tests parse YAML. This is CPU validation,
not the real training/resume pilot. The inherited historical
`requirements.txt` is NOT used.

Preview:

```bash
bash cluster_rpl/setup_environment.sh
```

The following creates one new environment and installs the
listed conda/pip dependencies. Run package-heavy preparation in an allocated
CPU context; do not compile packages on moria. It refuses an existing prefix:

```bash
bash cluster_rpl/setup_environment.sh --execute
```

Verify with `mamba run -n dd2414 python -m pip check` and environment export.
Existing environments are
inspected, never overwritten. Do not upgrade NVIDIA drivers or install a
system CUDA toolkit. Batch execution uses the mamba-created environment's
absolute Python path directly so checkpoint signals do not hit a wrapper.

The owner separately authorized removal of only
`/Midgard/home/hongen/miniforge3/envs/dd2414-hybrid`, after checking that no
job uses it and validating the replacement `envs/dd2414`. Old environment
manifests are preserved in `PROJECT/.cache/env_backups/`; they permit package
reconstruction, not instant restoration of a deleted environment. Cleanup job
`1250199` completed successfully; only base and the validated `dd2414` remain
registered. Base and other environments must remain untouched.

## 5. Configure and import only finished work

Create a personal config at `PROJECT/.cache/rpl.json`, based on
`config.example.json`. Choose a real node; the placeholder is
deliberately non-submittable. Store no credentials. Keep BF16 on a proven
native-BF16 node to match the workstation treatment; if FP16 is necessary,
declare it before starting and do not silently combine it with BF16 treatment
results as a fully matched comparison. Workers default to four, batch stays one.

On the workstation, snapshot only completed results:

```bash
python cluster_rpl/control.py export-completed --study /home/students2026/dd2414/fullsize_baseline/runs/official_rgb_full_10sweep_20261103 --output .cache/workstation_completed.json
```

Copy that small index and configure `completed_index` to its
absolute cluster path. If downstream tasks need a completed pretraining,
also copy its `summary.json`, `config.json`, and `checkpoints/latest.pt` into
`RUN_ROOT/EXACT_TASK_NAME/`. The checkpoint hash must match the exported index.
No checkpoint payloads/configs are rewritten. Do not copy active runs as finished.

The optional `--allow-source-relocation` changes ONLY how a downstream parent
compares the official RGB file's path. SHA-256, loaded tensor count, scope,
source settings, manifest, BEVCar digest and strict model tensor loading remain
checked. It does not relax same-run optimizer/sampler/RNG resume signatures.
Interrupted workstation runs are not claimed to resume exactly on a different
machine/precision/code version; finish them there or start a separately named
experiment after approval. Never patch checkpoint signatures to force a resume.

## 6. CPU audit, GPU pilot, then explicit single-task jobs

All command examples here assume the cluster project directory, personal
config `.cache/rpl.json`, and an appropriate allocated CPU context for auditing.
The routine workflow is authorized under the boundary described above.

The full stat-heavy audit must run on the data node, not moria:

```bash
srun /Midgard/home/hongen/miniforge3/envs/dd2414/bin/python cluster_rpl/check_assets.py --config .cache/rpl.json
```

It checks package versions, pinned teacher/source/RGB assets, official split
membership, manifest digest and EVERY `sample_data.json` reference for present,
nonempty sensor files. It writes `RUN_ROOT/assets_audit.json`. It is an existence
audit, not a substitute for rsync/checksum payload verification. Metadata can
consume several GiB RAM; the suggested CPU allocation is 32 GiB.

Print the dependency-aware inventory (no jobs submitted):

```bash
python cluster_rpl/control.py --config .cache/rpl.json plan
```

Preview a real three-update/full-manifest engineering pilot:

```bash
python cluster_rpl/control.py --config .cache/rpl.json submit --task official_rgb_double_dino_10sweep_velocity_zero_pretrain --pilot
```

To submit, use the same command with `--execute`. A scheduling-only check
uses `--test-only --execute` and does NOT replace the real pilot. The pilot
trains two updates, exits, starts a new process, resumes to update three and
requires finite loss and gradient norm. It validates operation, not numerical
identity across GPUs, convergence or generalization. Native BF16 and a single
Slurm-visible GPU are checked. OOM means stop and report; never silently lower
image resolution, sweep count, model capacity or dataset coverage.

Only after the audit and pilot pass, preview the first formal task:

```bash
python cluster_rpl/control.py --config .cache/rpl.json submit --task official_rgb_double_dino_10sweep_velocity_zero_pretrain
```

Inspect and log the exact printed `sbatch` command before `--execute`.
Submission creates log directories and a submission record, requests one GPU,
pins the data node, and runs exactly that task. A returned Job ID means queued,
not necessarily running. No dependent job is automatically submitted.
Inspect the outcome before selecting the next task from `plan`.

The wrapper requests an early checkpoint signal five minutes before the wall
limit. Partial exits are non-successful and do not unlock downstream work.
After inspection, resubmit the SAME task/config with
`--resume --execute`. There is no automatic requeue loop. Do not change epoch
budgets, precision, paths or code and bypass a resume mismatch. A pre-timeout
signal is not guaranteed to finish a long validation before the hard limit;
the last durable training checkpoint remains the recovery point.

## 7. Monitor and back up

The full dataset is currently uploading from the workstation to **arwen**.
It does not depend on a chat session remaining open. On the workstation:

```bash
systemctl --user status dd2414-rpl-nuscenes-upload.service
tail -F /home/students2026/dd2414/.cache/worktrees/rpl-full-data-double-distill/.cache/rpl_nuscenes_upload.log
```

The receiver is CPU-only Slurm job `1250189` (a restart can assign a new ID).
The service uses partial-file recovery and bounded retries; there is no
`--delete`. Source size is 419,984,351,322 bytes (391.14 GiB). Rsync's live
percentage uses its incrementally discovered file list, **not necessarily that
whole-dataset denominator**. Verify service exit, transfer totals and the full
asset audit before declaring the dataset complete or launching formal work.

Slurm was verified as `17.11.10`: use its supported basic options. In
particular, `srun --overlap` is unavailable; do not copy modern Slurm monitoring
flags without checking. Binary rsync transport requires `srun --unbuffered`.

Query `squeue -u hongen`, the returned Job ID with
`sacct -j JOB_ID`, or this lightweight progress renderer:

```bash
python cluster_rpl/control.py --config .cache/rpl.json progress
```

Each task has `status.json`, `summary.json`, `console.log`, `slurm-JOBID.out`,
`slurm-JOBID.err`, TensorBoard events and `checkpoints/latest.pt`. TensorBoard
and Jupyter must run inside an approved compute allocation, not moria. An
additional `srun --jobid=... nvidia-smi` monitoring step may wait for step
resources; do not interpret that alone as a stalled training job.
Cancel only a specifically approved Job ID. Never cancel all account jobs.
Periodically back up durable checkpoints/results to the workstation; local
storage and the shared NAS do not provide a guaranteed backup service.

## Local checks and current limitations

```bash
python -m unittest discover -s cluster_rpl -p test_cluster.py -v
PYTHONPATH=scripts python -m unittest scripts.test_official_full_data scripts.test_full_data_protocol -v
bash -n cluster_rpl/job.sh
bash -n cluster_rpl/setup_environment.sh
```

CPU contracts are not a CUDA training pilot. Environment installation and CPU
validation, node-local capacity, the real rsync route, and pinned RGB, DINO and
BEVCar assets have been verified. Full-data upload is still in progress.
The complete sensor audit, real three-update/resume pilot, full-model GPU memory
feasibility and formal cluster training remain pending. Synthetic CUDA job
`1250200` passed native-BF16 convolution forward/backward on an RTX 3080 with
9.64 GiB visible memory and released its GPU after 12 seconds; this does not
replace the real pilot. The one-GPU code path is intentional; there is no
DDP, shared-memory pooling or autonomous job scheduler.
