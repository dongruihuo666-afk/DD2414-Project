# Full-size epoch baseline

This directory isolates the new epoch-matched 4,096-versus-full experiment
from the historical fixed-update radar-scaling artifacts.

The locked comparison keeps the existing one-sweep model and objective while
training two independently initialized runs for eight complete data passes:

| Run | Training samples | Epochs |
| --- | ---: | ---: |
| `scale4096_seed125` | 4,096 | 8 |
| `full28130_seed125` | 28,130 | 8 |

The 4,096-sample control is complete: 6,560 finite updates, 32,768 sample
exposures, zero AMP overflow retries, and complete validation at epochs
1/3/5/8. Matched semantic/motion loss improved from `0.33595/0.91730` at epoch
1 to `0.30503/0.28907` at epoch 8. The full 28,130-sample run is the remaining
half of the comparison.

Runtime products are intentionally ignored by Git:

- `cache/`: frozen-teacher feature caches;
- `runs/<run>/checkpoints/`: resumable model states;
- `runs/<run>/metrics.jsonl`: batch-level training metrics;
- `runs/<run>/tensorboard/`: TensorBoard event files;
- `runs/<run>/status.json`: atomic current status for shell monitoring; and
- `runs/<run>/validation/`: per-frame validation records.

Compact configurations and final summaries may be committed separately after
they have been reviewed.

Preview the selected full-data command without starting it:

```bash
bash scripts/run_fullsize_baseline.sh
```

Select the 4,096 control with `SCALE=4096`. Execution remains opt-in through
`--execute`; the launcher defaults to the measured batch size 5, eight loader
workers, BF16, eight epochs, seed 125 and one radar sweep. Each run starts from
its own initialization and never resumes a historical fixed-update checkpoint.
The frozen configuration is batch 5, eight workers and BF16. A 256-update
stability pilot processed 1,280 unique samples at 12.27 samples/s, used 16.721
GiB peak PyTorch allocation (20,560/24,564 MiB total process memory at the
final status sample), and had no non-finite losses, gradients, or AMP overflow
retries. Batch 6 was only 1.6% faster in the earlier matched short pilot but
used 23,449 MiB, so it was rejected for insufficient memory headroom.

An opt-in synchronized phase profile (`--profile-timing`) measured the stable
batch-5 step as 8.3% frozen DINO extraction, 6.8% radar target projection,
14.0% input/motion-target construction, and 70.9% model forward/backward plus
AdamW. Removing DINO entirely would therefore have a theoretical throughput
ceiling of only about 14.05 samples/s, before cache I/O. The roughly 49 GiB
FP16 teacher cache is not used for this comparison: its complexity and storage
cost are not justified by that upper bound. The flag is for engineering pilots
only and adds CUDA synchronizations; do not use it for formal training.

The two formal launch commands for the frozen configuration are:

```bash
SCALE=4096 bash scripts/run_fullsize_baseline.sh --execute
bash scripts/run_fullsize_baseline.sh --execute
```

The post-baseline radar-density follow-up uses the same model, objective,
split, seed, optimizer and eight-epoch budget. Set `NSWEEPS=5` or
`NSWEEPS=10`; each setting receives an isolated run directory such as
`full28130_sweeps5_seed125`. A real data/voxel/forward/backward/resume smoke is
required before each formal multi-sweep launch. These runs retain the legacy
frozen random camera encoder and must not be described as the future
dual-level-distillation model.

```bash
NSWEEPS=5 bash scripts/run_fullsize_baseline.sh --execute
NSWEEPS=5 bash scripts/run_fullsize_baseline.sh --execute --resume
```

Append `--resume` to the same command after an interruption. Configuration
validation rejects a resume with a different batch size, optimizer, precision,
epoch count, manifest, or validation schedule.

## Monitoring

The trainer updates `status.json`, `metrics.jsonl`, and TensorBoard together.
For a concise terminal view:

```bash
watch -n 5 $HOME/miniconda3/envs/bev/bin/python \
  scripts/render_fullsize_progress.py \
  --run-dir fullsize_baseline/runs/full28130_seed125 --no-plot
```

The complete console stream is also retained automatically:

```bash
tail -F fullsize_baseline/runs/full28130_seed125/console.log
```

For a continuously refreshed plot that can remain open in the IDE:

```bash
MPLCONFIGDIR=.cache/matplotlib $HOME/miniconda3/envs/bev/bin/python \
  scripts/render_fullsize_progress.py \
  --run-dir fullsize_baseline/runs/full28130_seed125 --follow
```

This writes `progress.png` inside the run directory. For TensorBoard:

```bash
$HOME/miniconda3/envs/bev/bin/tensorboard \
  --logdir fullsize_baseline/runs --bind_all --port 6006
```

Open `http://localhost:6006` locally, or use SSH port forwarding when viewing
the remote host. GPU activity can be watched independently with:

```bash
nvidia-smi dmon -s pucvmet
```

These commands are monitors only; they do not start or alter training.

## Teacher-requested velocity control and downstream probe

The unattended follow-up keeps the same architecture, optimizer, seed,
one-sweep input, losses, eight-epoch budget, and full train/validation splits.
It compares one backbone trained with raw radar velocity against a separately
trained backbone in which only BEVCar input channels 4 and 5 (the two velocity
components) are zero throughout training and evaluation. Point locations,
RCS, masks, targets, tensor shapes, and model parameters are unchanged.

After epochs 1 and 8, each backbone is frozen and the same single `1x1`
convolution is trained for one epoch to predict the nuScenes vehicle BEV mask.
The formal result reports vehicle IoU, precision, recall, and F1 over all 6,019
validation samples. This is the requested test of whether explicit radar
velocity makes the learned BEV representation more useful. It is not a fully
trained end-to-end segmentation result: the historical camera encoder remains
frozen at random initialization, and only one seed is used.

The persistent pipeline runs these stages in order and stops on the first
failure: wait for raw-velocity training, run a two-update velocity-zeroed smoke,
run a four-sample probe smoke, train the velocity-zeroed backbone through epoch
1, run both epoch-1 probes, resume the velocity-zeroed backbone through epoch
8, then run both epoch-8 probes and the final comparison. Re-running the driver
skips completed stages and resumes from the latest exact checkpoint.

Monitor the entire chain with:

```bash
watch -n 10 $HOME/miniconda3/envs/bev/bin/python \
  scripts/render_teacher_overnight_progress.py
```

The service and its journal can be inspected with:

```bash
systemctl --user status dd2414-teacher-overnight.service
journalctl --user -u dd2414-teacher-overnight.service -f
```

Stage-specific logs and final Markdown comparisons are written below
`fullsize_baseline/teacher_overnight/`; probe checkpoints and metrics are below
`fullsize_baseline/probes/`. All runtime products are ignored by Git.
