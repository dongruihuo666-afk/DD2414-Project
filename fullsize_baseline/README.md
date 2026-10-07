# Full-size epoch baseline

This directory isolates the new epoch-matched 4,096-versus-full experiment
from the historical fixed-update radar-scaling artifacts.

The locked comparison keeps the existing one-sweep model and objective while
training two independently initialized runs for eight complete data passes:

| Run | Training samples | Epochs |
| --- | ---: | ---: |
| `scale4096_seed125` | 4,096 | 8 |
| `full28130_seed125` | 28,130 | 8 |

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
