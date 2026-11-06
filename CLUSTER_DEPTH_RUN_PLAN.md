# Isolated cluster depth-continuation study

Branch: `experiment/rpl-depth-10sweep-velocity-study`.
Workstation checkout, its eleven-task queue, and staged user PDF remain unchanged.

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

The current bounded change adds explicit completed-parent initialization and
an adjustable relative-depth teacher input resolution. It does NOT yet provide
the three-treatment Slurm matrix, admission audit extensions, or an automatic
queue. Existing cluster configuration still targets the old no-depth study.
Do not submit it as if it were this new depth study.

Before formal execution, implement the dedicated matrix and output root, copy
and hash-check the parent and Depth Anything source/weights, audit every sensor
reference on arwen, and pass real finite-update/new-process-resume tests for
all three depth modes. No silent resolution/sweep/model reductions after OOM.
The inherited low-resolution teacher size is not silently a formal choice:
lock the desired teacher resolution in the new matrix and record it explicitly.

## Resource authorization and current blocker

The supplied site manual explicitly restricts master-student accounts to one
GPU at a time. The owner subsequently requested trying two GPUs and setting
the manual aside. No administrator exception has been established. Scheduler
acceptance is not such an exception; no real two-GPU allocation is submitted.

On 2026-11-06 a non-allocating check on moria accepted:

```bash
sbatch --test-only --nodes=1 --nodelist=arwen --gres=gpu:2 --cpus-per-task=8 --mem=32G --time=00:10:00 --job-name=dd2414-two-gpu-feasibility --wrap=true
```

It printed a predicted start with an illustrative job number, but account
`squeue` was empty both before and after. Association `defaultusr` / QOS
`normal` showed no explicit MaxTRES/GrpTRES values in that query; this does not
prove exemption from site rules. Current trainer has no DDP implementation.
Even an authorized two-card allocation would not accelerate it automatically.

Next decision: obtain confirmation that the account has temporary two-GPU
permission (or that the supplied rule is obsolete), or keep the deployment
single-GPU. If two-GPU permission is confirmed, implement single-node DDP and
lock the new effective batch/epoch/update protocol before formal comparisons.
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
obtained. This must not be reported as a hardware failure or a successful test.
Do not retry through another wrapper, indirect execution or a different account.
The existing accepted scheduling-only test remains the sole two-GPU evidence.
Unblock a real test with confirmation of site authorization, not another user
approval to disregard the supplied manual.
