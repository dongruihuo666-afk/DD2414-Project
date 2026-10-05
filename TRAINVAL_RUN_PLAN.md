# nuScenes trainval review and run plan

Status: prepared on 2026-10-05 and subsequently executed after user approval.
The trainval data check, a one-step BEVCar VoxelNet smoke, the exact historical
64-train / 16-val / 512-step / three-seed / one-sweep reproduction, and a
uniform-scene sampling control completed. No all-frame training was started.
See the completed-results section below; remaining commands still preview unless
`--execute` is supplied.

## Repository and machine audit

- Fetched `origin`: local HEAD and `origin/dd2414-mini-baseline` both point to
  `3745647`; ahead/behind was 0/0. `origin/main` is still the upstream Simple-BEV
  commit `be46f0e`, 30 commits behind the feature branch. No pull was necessary.
- Existing staged changes from the earlier mini 64-sample / 5-/10-sweep work
  were present before this review and have been preserved. Equal branch heads
  do not mean those local experiment changes have already been published.
- Current environment: `${HOME}/miniconda3/envs/bev`, Python 3.10.14,
  PyTorch 2.5.1, torchvision 0.20.1, CUDA runtime 12.4. One RTX 3090 Ti with
  24,564 MiB VRAM, approximately 125 GiB system RAM and 376 GiB free disk.
- Full labeled nuScenes `v1.0-trainval` is installed at
  `${HOME}/datasets/nuscenes`. Mini was removed. Official split metadata gives
  700 training scenes / 28,130 keyframes and 150 validation scenes / 6,019
  keyframes, with no scene overlap. All 2,631,083 referenced sensor files were
  checked for existence and nonzero size: zero missing, zero empty. This is
  a filesystem/metadata audit, not a decode/checksum test of every sensor file.
- Frozen DINOv2 ViT-S/14 source and weights are cached in `.cache/torch`.
- Official BEVCar is checked out at commit
  `29cacda3bc5416d47428c1d0f017527acad34f90` under `external/BEVCar`.
  Project probes dynamically import only `nets/voxelnet.py`; no complete
  BEVCar model or BEVCar checkpoint is used.

## What the repository actually implements

| Route | Entry point | Supervision and scope |
| --- | --- | --- |
| Original Simple-BEV training | `train_nuscenes.py` -> `nets/segnet.py` | Human-box-derived segmentation/center/offset labels; streams batches and saves checkpoints |
| Original validation | `eval_nuscenes.py` | Iterates over the official validation split using a matching trained checkpoint |
| Dual-level semantic distillation | `scripts/eval_image_distill_heldout.py` | Frozen image DINOv2 and radar-anchored soft BEV targets; legacy metaradar student; 4 objective variants |
| BEVCar + motion/semantic experiment | `scripts/compare_motion_target_bevcar_heldout.py` | Official external VoxelNet radar encoder; semantic-only, motion-only, joint variants; no image-distillation head |
| Lightweight encoder and BEVCar bridge | `nets/radar_encoder.py`, `nets/bevcar_radar_bridge.py`, `nets/bevcar_voxel_adapter.py` | Opt-in experimental radar branches, with separate geometry/gradient checks |
| BEVFormer / Lift-Splat / other inherited networks | `nets/bevformernet*.py`, `nets/liftnet*.py`, etc. | Alternative upstream implementations; not selected by the current project launchers |

The dual-teacher probe trains the camera encoder from random initialization.
The BEVCar motion probe freezes a randomly initialized camera encoder; this is
an experimental control, not a mature pretrained camera backbone. Their losses
are not downstream segmentation accuracy. The existing loader still constructs
box-derived labels, although these two distillation objectives do not use them.

There is **no single completed trainer combining BEVCar, image distillation,
radar-guided fusion distillation, multi-sweep motion and a hybrid teacher**.
The soft Gaussian radar regions already exist. A new combined/hybrid architecture
still needs its own definition, integration and incremental ablations.

## Changes prepared in this revision

- Added `configs/trainval.env` and a default-preview launcher,
  `scripts/run_trainval.sh`, using this host's `bev` environment, trainval data,
  one GPU and distinct `artifacts/trainval` output storage.
- The shared teacher-data loader now accepts `dset=mini|trainval` and can return
  both splits sharing one nuScenes devkit instance. Historical mini callers
  retain their defaults.
- Both held-out probes accept `--dset`, `--sample-selection` and `--nsweeps`.
  Existing shell wrappers forward `DSET`, `SAMPLE_SELECTION`, and `NSWEEPS`.
  The motion probe previously forced one sweep.
- The trainval profile selects 64 training / 16 validation samples uniformly
  across the sorted official splits, rather than the first adjacent frames.
  Reports store dataset version, sweep count, exact indices, sample tokens and
  scene tokens. Split-scene overlap is checked before fetching sensor batches.
- A total-sample limit of 160 rejects accidental all-dataset GPU caching in
  these bounded probes. It is a guard, not a promise that every configuration
  below the limit fits on every GPU.
- The existing smoke helper accepts `--dset trainval`. The supervised trainer
  exposes `nworkers_val`; this profile sets both worker counts to zero because
  `VizData` currently allocates CUDA geometry while loading a sample.

Selecting trainval does **not** mean all 28,130 training frames are used by a
64-sample probe. All-frame self-supervised training cannot be enabled by changing
sample counts: the current probes retain every dense target and input on GPU.
FP16 `(384,200,200)` semantic targets alone for 34,149 frames need about 977 GiB,
without inputs, activations, optimizer state, or image targets. Neither VRAM nor
free disk can hold that dense cache. Those probes also lack periodic student /
optimizer checkpoints and resume support.

## How to preview or later execute

From the repository root, inspect a command without loading data or a model:

```bash
bash scripts/run_trainval.sh dual-teacher
NSWEEPS=10 bash scripts/run_trainval.sh motion
bash scripts/run_trainval.sh supervised
```

No interactive `conda activate` is needed: the launcher uses the configured
interpreter and activates the environment only for explicit execution. For an
interactive shell, use:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bev
```

Default preparation parameters:

| Setting | Held-out semantic / BEVCar motion probe | Supervised full-split baseline |
| --- | --- | --- |
| Dataset | trainval, official scene-disjoint splits | trainval, official scene-disjoint splits |
| Training / validation frames | 64 / 16; bounded uniformly spaced subsets | 28,130 / 6,019; training streams the split |
| Updates | 512 per variant and seed | 30,000 at batch 1 / accumulation 1 (~1.07 passes) |
| Seeds | 125, 42, 7 | Existing original trainer seed behavior |
| Radar history | 5 by default; compare 1, 5, 10 | 5 by default |
| Student image size | 112 x 192 | 224 x 400 (`RES_SCALE=1`) |
| Workers / GPUs | 0 / GPU 0 | train 0, val 0 / GPU 0 |
| Learning rate | 2e-4, existing probe objectives | 3e-4, original supervised objective |
| Checkpoints | Not yet supported by held-out probes | Every 1,000 updates; default retains latest |

`nsweeps` means up to that many historical radar sweeps **per radar sensor**;
all five radar sensors are already merged by the loader. Scene starts can have
less history. It does not add temporal camera attention. Motion multi-sweep
results require particular care because historical moving returns are merged
without object-motion compensation; compare one sweep before attributing gains.

## Proposed execution order (not run yet)

1. **Read one batch and perform one smoke update.** Check all modalities,
   finite loss/gradients, checkpoint round-trip, GPU memory and timing:

   ```bash
   bash scripts/run_trainval.sh data-check --execute
   bash scripts/run_trainval.sh smoke --execute
   ```

   The smoke is a low-resolution metaradar engineering test. Before a long
   supervised run, check its actual 224x400 / full-precision training path with
   a short run (disable the long-run scheduler for this five-step check):

   ```bash
   MAX_ITERS=5 SAVE_FREQ=5 USE_SCHEDULER=False EXP_NAME=trainval_supervised_smoke \
     bash scripts/run_trainval.sh supervised --execute
   ```

2. **Re-establish the semantic baseline on trainval subsets.** Keep the same
   selected frames, 512 updates, four objective variants, and three seeds;
   vary only radar history. Run sequentially to avoid VRAM contention:

   ```bash
   NSWEEPS=1 bash scripts/run_trainval.sh dual-teacher --execute
   NSWEEPS=5 bash scripts/run_trainval.sh dual-teacher --execute
   NSWEEPS=10 bash scripts/run_trainval.sh dual-teacher --execute
   ```

   Compare held-out fusion loss, image-feature similarity and no-radar penalty,
   including variability across seeds. Mini numbers are historical references;
   changing to trainval also changes frames/scenes and the selection protocol.
   The existing 0.8/0.2 objectives remain controls, not a claim that those weights
   are optimal. Sweeping 0.6/0.7/0.8 radar weights needs an additional objective
   configuration change; do it separately from dataset adaptation.

3. **Prepare and evaluate the BEVCar motion route independently.** Obtain the
   official checkout pinned by the project, then use the same frames/seeds:

   ```bash
   git clone https://github.com/robot-learning-freiburg/BEVCar.git external/BEVCar
   git -C external/BEVCar checkout --detach 29cacda3bc5416d47428c1d0f017527acad34f90
   NSWEEPS=1 bash scripts/run_trainval.sh motion --execute
   NSWEEPS=5 bash scripts/run_trainval.sh motion --execute
   NSWEEPS=10 bash scripts/run_trainval.sh motion --execute
   ```

   Record correct-radar, zero-velocity and no-radar losses. The current mean
   includes zero-coverage stationary frames: add moving-frame/cell-weighted
   summaries before interpreting a small velocity penalty. Add matched vs
   wrong-scene radar as a separate control. No BEVCar weights are downloaded
   or used for the label-free claim.

4. **Implement all-frame self-supervised training before scaling beyond the
   bounded probes.** Stream batches; cache only compact frozen image patch maps
   on CPU/disk (roughly 1.5 MiB/frame before compression, about 49 GiB for both
   splits), or extract online. Construct one batch's dense BEV target on demand.
   Add student/optimizer/scaler/RNG checkpoints, resume, epoch shuffling and
   separate periodic/full validation. Avoid generating unused box labels in
   pretraining. Validate memory stays bounded as sample counts grow. Start with
   one seed and one objective; scale to all 28,130 frames only after the smoke
   and timing checks. The current launcher does not implement this trainer.

5. **Integrate the requested combination one component at a time.** Fix the
   BEVCar backbone, sample manifest and training budget; add image distillation,
   radar-majority weighting, soft-region controls, then the explicitly defined
   hybrid teacher. Compare each change against the preceding baseline. Plan
   downstream labeled evaluation separately; feature loss is not vehicle IoU.

An independent supervised full-split baseline is already supported by the
original streaming trainer and the new profile:

```bash
bash scripts/run_trainval.sh supervised --execute
INIT_DIR=/absolute/path/to/the/generated/checkpoint-directory \
  bash scripts/run_trainval.sh eval --execute
```

This uses human labels and legacy metaradar, not the combined self-supervised
BEVCar model. Its in-training validation evaluates one batch every 100 updates;
run the separate `eval` command for all 6,019 validation frames. Keep evaluation
resolution, encoder and sweeps matched to training. The 30,000-update preparation
budget covers about 1.07 passes over the training split; it is a starting
baseline, not a reproduction of the original four-GPU paper budget or a guarantee
of convergence. Step 30,000 is saved with the default 1,000-step save interval.

## Completed trainval reproduction (2026-10-05)

After the user approved execution, the one-batch one-sweep data check loaded
28,130 train and 6,019 validation samples and 313 valid radar points from the
first train frame. A one-train / one-val / one-step `motion_only` smoke completed
the frozen-DINO target, VoxelNet, fusion, backward and ablation paths.

The strict one-variable reproduction used the previous experiment's 64/16
sample counts, head selection, 512 updates, seeds 125/42/7, one radar sweep,
learning rate 2e-4 and semantic-only/motion-only/joint variants. Only the
nuScenes version changed from mini to trainval:

```bash
TRAIN_SAMPLES=64 VAL_SAMPLES=16 STEPS=512 SEED_LIST=125,42,7 \
  NSWEEPS=1 SAMPLE_SELECTION=head \
  TAG=trainval_train64_val16_nsweeps1_head \
  bash scripts/run_trainval.sh motion --execute
```

| Selection / variant | Motion correct | Zero velocity | Penalty | No radar |
| --- | ---: | ---: | ---: | ---: |
| head, semantic-only | 1.0922 | 1.0920 | -0.0002 | 1.0760 |
| head, motion-only | 1.0641 | 1.1409 | +0.0768 | 0.9097 |
| head, joint | 1.2210 | 1.3014 | +0.0804 | 0.9473 |
| uniform, semantic-only | 2.5432 | 2.5407 | -0.0026 | 2.5674 |
| uniform, motion-only | 1.6998 | 2.4977 | +0.7980 | 2.6245 |
| uniform, joint | 1.5796 | 2.4398 | +0.8602 | 2.6140 |

The strict head subset covered only 2 training scenes and 1 validation scene;
6/16 validation frames had no moving target cells. Its moving-frame-only
motion penalties were +0.1228 (motion-only) and +0.1286 (joint), but removing
all radar lowered rather than raised loss. It is the exact protocol reproduction
but weak evidence of generalization. The uniform control covered 64/16 distinct
train/val scenes; 15/16 validation frames had moving targets. Its moving-only
penalties were +0.8512 and +0.9176, and no-radar loss was higher than correct.
This establishes strong VoxelNet velocity sensitivity for the broader sampled
control, not a full 28,130-frame training result.

Reports:

- `artifacts/trainval/motion_target_bevcar_heldout_trainval_train64_val16_nsweeps1_head.json`
- `artifacts/trainval/motion_target_bevcar_heldout_trainval_train64_val16_nsweeps1_uniform.json`

## Verification boundary and next action

The configuration changes passed Python compilation, shell syntax, mocked
routing/selection tests and launcher previews. The executed checks above add a
real trainval batch, VoxelNet forward/backward, target generation and both
64/16 experiments. Full-split self-supervised training, 5/10-sweep VoxelNet
comparisons, checkpoint/resume behavior and downstream IoU remain untested.
The next bounded choice is whether to repeat the broader control at 5/10 sweeps
or implement streaming all-frame self-supervised training.
