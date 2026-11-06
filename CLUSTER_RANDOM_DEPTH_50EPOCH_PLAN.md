# Random ResNet101, double DINO distillation, and projected metric depth

Branch: `experiment/rpl-depth-10sweep-velocity-study`. This replaces the earlier
three-way, five-epoch warm-start continuation for the NEW cluster experiment.
The workstation branch/queue and official Simple-BEV path are unchanged.

## Two matched experiments

| Setting | DINO-only control | DINO + metric depth experiment |
| --- | --- | --- |
| Student initialization | Entire student random, seed 125 | Identical initial student tensors |
| Camera / radar | ResNet101 / BEVCar VoxelNet | Same |
| Inputs | RGB 448x800; 10 radar sweeps; velocity provided | Same |
| Camera geometry | 80-bin softmax depth head, 1–80 m; probability-weighted lifting | Same |
| Frozen DINOv2 ViT-S/14 | Image and radar-supported fused BEV distillation | Same |
| Frozen metric Depth Anything V2 Small | Absent | Outdoor VKITTI metric teacher, max depth 80 m |
| Pretraining loss | 0.2 image DINO + 0.8 fused BEV DINO | Same + 0.05 projected depth cross entropy |
| Extra motion / radar depth losses | Absent | Absent |
| Budget | **50 complete training epochs** | **50 complete training epochs** |

There is no official RGB encoder, ImageNet initialization, previous student
checkpoint, or velocity-hidden control in this study. Pretrained frozen
teachers are distinct from the randomly initialized trainable student. The
student segmentation decoder is not trained during distillation; it is trained
in the later supervised segmentation evaluation.

The two students have the SAME depth-aware architecture. This isolates the
additional metric teacher supervision. It does not isolate the architecture
change against the older depth-free projector; historical results are not this
experiment's matched control.

## How depth changes the projection

1. ResNet image features feed a single 1x1 convolution (128 to 80 channels).
2. A softmax predicts probabilities for metric depth bins along each image ray.
3. Existing intrinsics/extrinsics map BEV voxel centers to image coordinates
   and camera-Z. Each image contribution is weighted by its depth probability
   at that voxel. This is applied before camera aggregation/radar fusion.
4. In the depth experiment, frozen metric teacher depths become soft target
   distributions (Gaussian width 2 m). Sample these targets and student log
   probabilities into the SAME calibrated camera BEV volume. Compute a
   per-camera, target-mass-normalized cross entropy there, before radar fusion.

A target near 20 m rewards probability near 20 m; softmax competition reduces
mass at alternative distances on that ray. The loss is attached after geometric
projection, rather than comparing arbitrary scalar hints replicated along rays.
Its gradients reach the depth head and ResNet. The student probability head
remains at inference; both frozen teachers can be discarded.

The metric teacher uses `Depth-Anything-V2/metric_depth`, not the relative-depth
module with renamed weights. Official outdoor Small weight SHA-256:
`9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb`.
Source commit remains `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
See the [official metric model instructions](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).
Teacher outputs are pseudo depth with possible synthetic-to-nuScenes scale or
domain error. They are not independent dense ground truth or occupancy labels.
The 1–80 m support is an explicit limit; voxels outside it receive no camera
contribution. Shared legacy XY sampling is preserved; the experimental depth
axis uses correct `align_corners=False` bin-center coordinates.

## Epochs, optimization, resources, and monitoring

Use the official 700-scene training split (28,130 keyframes) and 150-scene
validation split (6,019). Each epoch includes every training keyframe exactly
once. The 50-epoch budget is **1,406,500 training sample visits per model**, not
50 small subsets. Validation never updates parameters, and segmentation labels
are used only by downstream supervised heads/adaptation. This is the established
official train/validation protocol; validation is not a new blind test set.

Both groups use AdamW at 2e-4 for ALL trainable student components, weight decay
1e-5, one full epoch of warmup, cosine decay to 10% of the initial rate, clipping
at 5, and native BF16. No early stopping changes one group's training exposure.
Fifty epochs are a first budget, not a convergence guarantee. Compare the DINO
terms separately across groups; the treatment's extra CE makes its TOTAL loss
incomparable to the control's total loss. Pretraining losses are not Vehicle IoU.

| GPUs per model | Global batch (1/rank) | Updates/epoch | Total updates | CPUs | RAM candidate |
| --- | --- | --- | --- | --- | --- |
| 2 | 2 | 14,065 | 703,250 | 8 | 32 GiB |
| **5 (initial candidate)** | **5** | **5,626** | **281,300** | **20** | **80 GiB** |
| 10 | 10 | 2,813 | 140,650 | 40 | 160 GiB |

These world sizes divide the full split exactly. Four/eight ranks are rejected
instead of padding or dropping samples. Both treatments must use the SAME world
size/global batch. More GPUs are not a promise of linear speedup. Start with
finite-update/memory/resume pilots at the candidate size, then measure sustained
throughput before locking a formal run. A world-size change requires a fresh
matched pair; exact resume does not permit it. Per-rank BatchNorm is retained,
with rank-zero buffer broadcast; this is not SyncBatchNorm or single-GPU
numerical equivalence.

Two independent five-GPU jobs can run concurrently if arwen has all ten GPUs,
40 CPUs and 160 GiB available. Jobs request their own resources through Slurm;
do not manually set visibility to GPUs assigned to another job. The latest
read-only resource check found arwen's ten GPUs allocated, so no immediate start
or finish time is promised. Data are local to arwen; do not change nodes without
staging/validating the data there first.

Training writes `status.json`, per-update `metrics.jsonl`, and TensorBoard, plus
`validation/epochNNN/summary.json` after EVERY full epoch. Latest checkpoints are
atomic, periodic (default every 250 updates) and saved on the graceful stop flag.
Full validated milestones are retained at epochs 5/10/20/30/40/50, including when
resuming directly into unfinished epoch-end validation. Each checkpoint includes
optimizer, exact sampler cursor, per-rank RNG/scaler/BatchNorm buffers and source
hashes. Only rank zero writes. Interrupted validation is rerun rather than
mistaken for a completed epoch evaluation.

## Deployment and bounded launch procedure

These commands are instructions, not evidence that a job was submitted. Run
heavy audits, teacher inference and training ONLY in Slurm compute allocations.
Keep the environment managed by the existing Miniforge/mamba installation.

On moria, update a clean checkout without overwriting local changes:

```bash
cd /Midgard/home/hongen/projects/DD2414-Project
git status --short --branch
git fetch origin
git switch experiment/rpl-depth-10sweep-velocity-study
git pull --ff-only origin experiment/rpl-depth-10sweep-velocity-study
cp cluster_rpl/random_depth_study.example.json .cache/random_depth_50epochs.json
```

Prepare the metric weight on the workstation and transfer the resulting file
to the config's exact `metric_teacher_weights` location. Never replace the old
relative-depth weight: the studies must remain distinguishable. Download:

```bash
curl --fail --location --retry 2 --output .cache/depth_anything_v2_metric_vkitti_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
sha256sum .cache/depth_anything_v2_metric_vkitti_vits.pth
```

No student checkpoint transfer is needed. Existing DINO cache, pinned BEVCar
source and Depth Anything source remain necessary. No system-wide installation
or environment deletion is performed by these launchers. The target mamba prefix
is `/Midgard/home/hongen/miniforge3/envs/dd2414` (Python 3.10, torch 2.5.1+cu121,
torchvision 0.20.1, NumPy 1.23.5, Pillow 9.4, nuScenes SDK 1.1.11).

Preview both plans (stdlib only; no compute/allocation):

```bash
python cluster_rpl/random_depth_study.py --config .cache/random_depth_50epochs.json
```

Audit all official scene/sample/log memberships and all 2,631,083 metadata sensor
references, plus pinned teacher/source hashes. This audit deliberately does NOT
require an official RGB encoder or old parent. Its report must match the exact
commit/config before the launcher can execute:

```bash
sbatch --job-name=dd2414-random-depth-audit --nodelist=arwen \
  --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:20:00 \
  --chdir=/Midgard/home/hongen/projects/DD2414-Project \
  --output=.cache/random-depth-audit-%j.out --error=.cache/random-depth-audit-%j.err \
  --wrap='/Midgard/home/hongen/miniforge3/envs/dd2414/bin/python cluster_rpl/audit_random_depth.py --config .cache/random_depth_50epochs.json'
```

After the audit passes, submit ONE short five-GPU pilot using the ACTUAL batch
wrapper (repeat for each treatment). Do not introduce a small dataset/subset;
stop after two updates on the real full-data sampler:

```bash
sbatch --job-name=dd2414-dino-only-pilot --nodelist=arwen \
  --ntasks=1 --gres=gpu:5 --cpus-per-task=20 --mem=80G --time=00:30:00 \
  --signal=B:USR1@300 --chdir=/Midgard/home/hongen/projects/DD2414-Project \
  --output=.cache/random-dino-only-%j.out --error=.cache/random-dino-only-%j.err \
  cluster_rpl/random_depth_job.sh \
  /Midgard/home/hongen/projects/DD2414-Project/.cache/random_depth_50epochs.json \
  dino-only --stop-after-updates 2
```

The intentional stop returns **75** with `summary.status=partial`; this is NOT
successful completion or a segmentation result. Launch the same wrapper again
with `--resume --stop-after-updates 3` to verify a fresh-process restore. Repeat
for `dino-metric-depth-bev`. Check finite loss/gradients, unchanged scientific
config, update 3, 15 committed sample visits for five ranks, every rank's cursor
and RNG, and real `nvidia-smi` memory. No automatic OOM fallback silently changes
one treatment. Pilots start the SAME 50-epoch trajectory; resume preserves that
budget, rather than adding 50 more epochs.

After both pilots pass, the formal command is the same per-treatment wrapper
with `--time=47:00:00`, `--resume`, and WITHOUT `--stop-after-updates`. Do not
keep a 30-minute pilot time limit for the long job. Run the two treatments as
independent Slurm jobs if resources permit. Never submit another writer for a
run directory that is already active; launcher/trainer locks also reject this.

Fifty epochs may exceed one 47-hour allocation. `B:USR1@300` requests a safe
collective checkpoint; a partial exit 75 must be resumed in another allocation
with the SAME commit/config/world size. There is no implicit requeue or indefinite
job-submission loop. Preserve checkpoints/status on durable shared storage at
safe boundaries; arwen local storage is not a backup. Time estimates require a
sustained pilot, including epoch validation/checkpoint overhead.

Monitor with `squeue -u hongen`, `tail -F .cache/random-dino-only-JOBID.out`,
the treatment's `status.json`, and `srun --jobid=JOBID nvidia-smi` inside the
existing allocation. TensorBoard also requires a compute allocation.

## Semantic evaluation after pretraining

The preview lists three downstream commands per treatment, all using the FULL
official train/validation split: frozen camera-BEV 1x1 probe (10 epochs), frozen
fused-BEV 1x1 probe (10 epochs), and fully trainable segmentation with the
Simple-BEV decoder (5 epochs). The two probes have the same 128-channel input
and one 1x1 readout; camera features use fixed height averaging. Decoder/readout
capacity differs from a frozen probe and is reported separately.

Downstream remains single-GPU in this implementation. It strictly loads the
new 80-channel-head checkpoint/model profile and requires all 50 pretraining
epochs AND full final validation. Frozen teachers are unnecessary downstream.
Main metric: GT vehicle BEV IoU at the official threshold/valid mask, with
precision/recall/F1 as diagnostics. Before/after-fusion probes and final
segmentation answer different questions. Teacher losses remain auxiliary;
compare against official Simple-BEV references with initialization/budget and
radar encoder differences explicitly stated.

## Actual implementation checks

131 CPU regression tests pass, including real two-process Gloo gradients for
both new forward paths, legacy objective equivalence, fresh-process checkpoint
resume and relocated Slurm spool execution. Calibration tests distinguish
10/20/30 m on one ray and verify camera gradients and extrinsic transforms.
The actual student factories produce identical random states with all checkpoint
loading forbidden (only inherited coordinate allocation redirected to CPU).
Official metric weights strictly load and produce finite outputs for two
synthetic CPU images; teachers remain frozen. Local official metadata membership
and scene/sample/log disjointness pass. See the
[implementation evidence](artifacts/random_metric_bev_implementation_20261106.json).

These checks are NOT a full-model CUDA pilot, measured multi-GPU throughput,
convergence or semantic/depth accuracy. No new cluster job was submitted by this
implementation task. The next execution gate is the allocated CPU audit and
finite-update/resume pilots described above.
