# Isolated cluster depth-continuation study

Branch: `experiment/rpl-depth-10sweep-velocity-study`.
Workstation checkout, its eleven-task queue, and staged user PDF remain unchanged.

## Latest runtime checkpoint — 2026-11-06

The owner authorized deployment and short admission trials. Full data/asset
audit `1250328` passed. Full-model two-GPU pilots `1250329` (no depth), `1250330`
(radar depth) and `1250331` (radar + Depth Anything V2) all completed normally
after updates one/two and a NEW-process resume to update three. GPUs were
released, and no formal five-epoch phase has started. See the compact record
`artifacts/rpl_depth_admission_20261106.json`; this is actual CUDA execution/
restore evidence, not convergence, full validation or semantic accuracy.

Peak rank-zero PyTorch memory observed was 6.83 / 7.80 / 7.89 GiB. The Depth
Anything case had large pre-clipping gradient norms (557/768/305) under the
unchanged clip-at-five policy; monitor this in longer training. A telemetry-only
fix corrects stale learning-rate reporting after optimizer restore. Checkpoint
inspection confirms actual optimizer warmup rates were correct in all pilots.
Strict source-digest resume guards remain unchanged, so old-code pilot
checkpoints are preserved as evidence, not silently resumed with changed code.

Before formal execution, confirm longer two-GPU authorization. Use a NEW
formal `run_root` (e.g. personal `dd2414_runs/depth_continuation_formal_20261106`)
in an ignored config, rerun CPU admission, and start all three fresh equal-
budget phases from the same approved parent. Do not overwrite or delete the
existing admission outputs. The three-step pilots are not counted as a full
training result. No automatic long-job queue/requeue has been installed.

## Approved scientific design

On 2026-11-06 the owner selected the completed, velocity-present, five-epoch
double-distillation checkpoint as the common initialization for three NEW
equal-budget phases: no depth; radar-supervised depth with depth-aware lifting;
radar depth plus a frozen Depth Anything V2 relative-depth teacher. Each phase
adds five full pretraining epochs with a fresh optimizer/sampler/schedule.
The no-depth control must also receive those five additional epochs: reusing
its earlier 51.09% downstream score is not an equal-budget comparison.

Pinned parent checkpoint SHA-256:
`ae9daf1124a03ee5a47f5fac68886334bcdd12dcbe5684c8a64167982c632520`.
Parent: `official_rgb_double_dino_10sweep_velocity_full_pretrain`, not the
supervised end-to-end segmentation checkpoint. It includes all model tensors,
not just the official RGB encoder. New phases load model tensors strictly;
they do not pretend to restore the parent's optimizer or exact sampler state.

Retain the official 28,130/6,019 train/validation frames, 448x800 RGB, 448x798
DINO, ten radar sweeps, velocity input present, seed 125, both DINO losses and
the unchanged motion auxiliary objective. Sparse radar depth is not dense GT;
the relative-depth teacher is not a metric-depth GT source. Depth loss and
lifting change together, so their individual effects are not isolated.

For each completed continuation, independently train end-to-end segmentation
(five epochs), before/after-fusion frozen 1x1 probes (ten epochs each), and a
GT-box-conditioned object-velocity readout (five epochs). Vehicle BEV IoU is
primary; precision/recall/F1 and velocity EPE are supplementary. Each downstream
branch starts from its own completed continuation, never another downstream
model. Scores are official-validation results, not blind-test results.

The matching velocity-hidden depth matrix is deferred until its five-epoch
parent completes on the workstation, its full validation/provenance is checked,
and an immutable checkpoint/hash is transferred. Never use a partial parent or
substitute velocity-present weights for that future comparison.

## Implementation and deployment gates

The branch now implements opt-in two-process DDP continuation pretraining,
the three-treatment command matrix and a signal-safe Slurm batch wrapper.
The initial implementation was LOCAL CODE PREPARATION only. The later approved
deployment and bounded CUDA results are recorded in the latest checkpoint
above; they supersede the earlier no-deployment restriction for admission.
The inherited `control.py` configuration still targets the old single-GPU
no-depth study; do not submit it as if it were this new matrix.

Each GPU processes one frame (six cameras). The global batch is two, with
14,065 synchronized optimizer updates per complete epoch / 70,325 per new
five-epoch phase. Every one of the 28,130 training frames is seen exactly once
per epoch, with disjoint shards and no padding/drop-last. Both validation
shards together score all 6,019 frames once (3,010 + 3,009), using the same
rank-zero model buffers. Training averages the two rank-local per-sample
objectives, not a global-cell-count-normalized loss. Learning rates stay
1e-5 / 2e-4 for camera / other student parameters; AdamW, quarter-epoch warmup
and cosine decay remain. All three treatments use this SAME new recipe.
It is not an identical optimizer trajectory or update budget to global-batch-
one historical runs. Memory is not pooled and a twofold speedup is not promised.

DINO and optional Depth Anything V2 stay frozen in separate per-rank copies.
The relative-depth teacher is explicitly 448x798 in this matrix. Trainable
student forward/projection heads execute inside DDP. BatchNorm uses rank-local
statistics, with rank-zero buffers broadcast before training forwards (not
SyncBatchNorm). The opt-in sample-local augmentation RNG is keyed by seed,
epoch and index, independent of worker prefetch; it is not the old worker-RNG
trajectory. Training RNG starts at seed-plus-rank so dropout masks need not
match across different local examples. Same-world-size resume restores each rank's sampler,
RNG, scaler and buffers, shared model/optimizer and global progress. A completed
single-GPU parent is model-only INITIALIZATION, never a two-GPU exact resume.

Downstream segmentation, both frozen probes and GT-box motion remain
single-GPU. Each treatment's four downstream commands start from its own
completed DDP continuation with unprefixed model keys. The matrix is three
pretraining tasks plus twelve readout tasks, NOT fifteen simultaneous jobs.
There is no automatic dispatcher, cross-host reservation or auto-requeue.

Before formal execution, confirm permission for longer two-GPU resource use
and select fresh formal output directories after the telemetry fix. Asset/
sensor audits and bounded full-model finite-update/new-process-resume tests
for all three modes are now passed. Their small scope does not establish
long-run stability/throughput or memory fit for every sample. No silent
resolution/sweep/model reductions after OOM. Automatic dispatch is future work.

The owner subsequently authorized deployment and bounded admission. The exact
parent/teacher assets have now been copied and CPU audit job `1250328` reported
passed hashes, official split membership and all 2,631,083 nonempty sensor-file
references in approximately 96 seconds (arwen, zero GPUs). This is presence/
provenance verification, not a byte-for-byte full-dataset checksum or a CUDA
training result. A long-running two-GPU allocation is not yet submitted.

Dedicated admission commands (inside their respective authorized allocations):

```bash
python cluster_rpl/check_depth_assets.py --config .cache/depth_study.json
python cluster_rpl/depth_pilot.py --config .cache/depth_study.json --treatment none
```

The pilot requires the passed audit and a fresh treatment output; it launches
two updates, saves both rank states, starts a NEW torchrun process, completes
update three and verifies sampler/global progress, finite checkpoint tensors,
changed model tensors and no replayed/skipped metrics. It writes
`admission_none.json`; repeat independently for `radar` and `radar-dav2`.
It never submits another job or starts formal training. The preserved
`pilot_update002.pt` is a local recovery/verification artifact, not a result.

### Local preview (no training or allocation)

```bash
python cluster_rpl/depth_study.py --config cluster_rpl/depth_study.example.json
```

This prints the resource profile and all fifteen exact commands. Copy the
example to an ignored `.cache/` config only when preparing actual deployment.
The parent path in the example is a future destination, NOT a transferred file.
Inspect/verify every asset and the configured node-local output root first.

### Future submission template (NOT executed in this update)

After deployment, authorization and admission pilots, inside the new branch
on moria, with an existing absolute config path:

```bash
sbatch --nodes=1 --ntasks=1 --nodelist=arwen --gres=gpu:2 \
  --cpus-per-task=8 --mem=32G --time=47:00:00 --signal=B:USR1@300 \
  --job-name=dd2414-depth-none \
  --output=/Midgard/home/hongen/projects/DD2414-Project/.cache/depth-none-%j.out \
  --error=/Midgard/home/hongen/projects/DD2414-Project/.cache/depth-none-%j.err \
  cluster_rpl/depth_ddp_job.sh \
  /Midgard/home/hongen/projects/DD2414-Project/.cache/depth_study.json none
```

Use `radar` and `radar-dav2` in place of `none` for the other treatments, one
explicit submission at a time under the authorized account limits. The wrapper
never submits anything itself and refuses execution on moria/outside Slurm.
It launches one torchrun agent with two local training processes. It holds an
output-specific launch lock, and rank zero also locks its checkpoint directory.
Use `--signal=B:USR1@300`, not a signal directly to torchrun: the wrapper writes
`RUN_DIR/REQUEST_CHECKPOINT`, both ranks agree at a safe boundary, save, and exit
partial. Partial exits return 75, not dependency-success. SIGKILL/node failure
cannot save a fresh checkpoint; resume from the latest atomic completed save.

For the real admission pilot, use a bounded allocation and append
`--stop-after-updates 2`; then start a NEW process with `--resume
--stop-after-updates 3` in that same treatment run directory. This uses full
manifest/model/resolution, not a sample subset. Once the pilot is checked,
resume with only `--resume` and a formal time limit. Code/configuration and world
size must not change between these resumes. Never automatically migrate a
single-GPU optimizer or resume a world-size-two run on one GPU.

Inspect `status.json`, `metrics.jsonl`, TensorBoard and `summary.json` per
treatment; pretraining is complete only when summary status is `complete`,
epoch five and all 6,019 final validation frames are present. Only then run
the printed downstream commands, each in a separate ONE-GPU allocation.
The primary result remains Simple-BEV-aligned Vehicle BEV IoU, not pretraining
loss. Validation radar-depth MAE is at radar-supervised anchors, not dense GT.

## Resource authorization and diagnostic history

**Current update:** after the initial rejection described below, the owner
explicitly confirmed administrator permission for the temporary two-GPU test.
A real test then passed as job `1250320`; see the final subsection. Permission
was scoped to that diagnostic, not presumed to cover formal multi-GPU training.

The supplied site manual explicitly restricts master-student accounts to one
GPU at a time. Initially, the owner requested trying two GPUs and setting
the manual aside, before an administrator exception had been confirmed.
Scheduler acceptance was not such an exception; no real allocation was made
at that preparation stage.

On 2026-11-06 a non-allocating check on moria accepted:

```bash
sbatch --test-only --nodes=1 --nodelist=arwen --gres=gpu:2 --cpus-per-task=8 --mem=32G --time=00:10:00 --job-name=dd2414-two-gpu-feasibility --wrap=true
```

It printed a predicted start with an illustrative job number, but account
`squeue` was empty both before and after. Association `defaultusr` / QOS
`normal` showed no explicit MaxTRES/GrpTRES values in that query; this does not
prove exemption from site rules. At that time the trainer had no DDP
implementation, so allocation alone could not accelerate it.

The next decision at that stage was to obtain temporary permission or retain
single-GPU execution. Permission was subsequently confirmed for the short
diagnostic below. Before formal DDP training, confirm that its longer resource
use is covered, and lock the effective batch/epoch/update protocol.
Do not alter the running workstation job or silently migrate its optimizer
state to a different world size.

### Bounded real-test request and execution rejection

The owner subsequently requested a temporary real two-GPU test. A synthetic
CUDA/NCCL/tiny-DDP diagnostic was prepared in ignored `.cache/` and uploaded to
the account's project cache. The proposed allocation was arwen, two GPUs,
four CPUs, 8 GiB RAM and a three-minute hard limit, with no dataset/checkpoint
access and no full-model training. The execution environment's automatic risk
review rejected the submission BEFORE the SSH/Slurm command executed because
of the verified site single-GPU rule and missing administrator exception.

No real job ID was issued, no GPUs were allocated, and no CUDA/NCCL result was
obtained in that rejected attempt. It was not a hardware failure or a successful test.
Do not retry through another wrapper, indirect execution or a different account.
The scheduling-only test was then the sole two-GPU evidence. Confirmation of
site authorization, rather than approval to disregard the manual, subsequently
unblocked the bounded real test below.

### Authorized real two-GPU result

After the owner's explicit confirmation that permission had been obtained, the
same bounded request was resubmitted through normal Slurm, with the new
authorization stated in the execution approval. Job `1250320` allocated
`gres/gpu=2`, four CPUs and 8 GiB on arwen, with a three-minute hard limit.
It completed in **16 seconds**, state `COMPLETED`, exit code `0:0`.
The account queue was empty afterward; both GPUs were released normally.

Both ranks saw distinct RTX 3080 devices (9.64 GiB each), torch 2.5.1+cu121,
and native BF16. NCCL all-reduce returned 3 on both ranks from inputs 1 and 2.
A tiny DDP linear model completed finite BF16 forward/backward/SGD, and a
post-update parameter-sum check agreed between ranks. The log ended with
`TWO_GPU_PROBE_PASSED`. The ranks' individual losses differed because they
used different inputs; synchronized updates, not identical local losses,
were the checked property. This small check is not an exhaustive tensor audit.

Cluster log (runtime-only):
`/Midgard/home/hongen/projects/DD2414-Project/.cache/rpl_two_gpu_probe-1250320.log`.
The tested script was the ignored cache copy prepared before permission;
`cluster_rpl/two_gpu_probe.py` preserves its test logic and additionally refuses
execution outside a Slurm allocation. It does not load project data or weights.

To reproduce within another specifically authorized two-GPU allocation:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=WARN \
  /Midgard/home/hongen/miniforge3/envs/dd2414/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 cluster_rpl/two_gpu_probe.py
```

Retain the explicit three-minute Slurm limit. Do not run this on moria or use
the temporary-test permission as an ongoing allocation exemption. At the time
of this diagnostic the actual research trainer remained single-GPU. The later
initially local-only implementation described above passed CPU/Gloo sampling,
gradient and resume checks. Subsequent owner-approved full-model CUDA pilots
now pass in all three modes (latest checkpoint above). Sustained speedup,
long-run stability and memory fit across every sample remain unmeasured.
