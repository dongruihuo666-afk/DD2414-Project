#!/usr/bin/env python3
"""Render and optionally follow one full-size training run."""

import argparse
import json
import math
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--refresh-seconds', type=float, default=15.0)
    parser.add_argument('--no-plot', action='store_true')
    return parser.parse_args()


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def read_status(run_dir):
    path = run_dir / 'status.json'
    if not path.is_file():
        raise FileNotFoundError(f'training status does not exist yet: {path}')
    return json.loads(path.read_text())


def read_metrics(run_dir):
    path = run_dir / 'metrics.jsonl'
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrently appended final line can be temporarily incomplete.
            continue
    return records


def terminal_summary(status):
    latest = status.get('latest') or {}
    ema = status.get('ema') or {}
    gpu = status.get('gpu') or {}
    message = status.get('message')
    lines = [
        f"state={status['state']} scale={status['scale']} "
        f"epoch={status['epoch_progress']:.3f}/{status['epochs']} "
        f"complete={status['percent_complete']:.2f}%",
        f"step={status['optimizer_step']}/{status['total_optimizer_steps']} "
        f"samples={status['samples_seen']}/{status['total_sample_exposures']} "
        f"speed={status['samples_per_second']:.2f} samples/s "
        f"ETA={duration(status.get('eta_seconds'))}",
        f"loss={latest.get('loss', float('nan')):.4f} "
        f"semantic={latest.get('semantic_loss', float('nan')):.4f} "
        f"motion={latest.get('motion_loss', float('nan')):.4f} "
        f"EMA={ema.get('loss', float('nan')):.4f} "
        f"lr={status['learning_rate']:.3g}",
    ]
    if gpu:
        lines.append(
            f"GPU={gpu['utilization_percent']:.0f}% "
            f"VRAM={gpu['memory_used_mib']:.0f}/{gpu['memory_total_mib']:.0f} MiB "
            f"temp={gpu['temperature_c']:.0f} C power={gpu['power_w']:.0f} W"
        )
    if message:
        lines.append(f'message={message}')
    return '\n'.join(lines)


def render_plot(run_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = read_metrics(run_dir)
    if not records:
        return False
    epochs = [record['epoch_progress'] for record in records]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [r['loss'] for r in records], alpha=0.18, label='batch')
    axes[0, 0].plot(epochs, [r['ema_loss'] for r in records], label='EMA')
    axes[0, 0].set_title('Joint loss')
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [r['semantic_loss'] for r in records], label='semantic')
    axes[0, 1].plot(epochs, [r['motion_loss'] for r in records], label='motion')
    axes[0, 1].set_title('Objective components')
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [r['learning_rate'] for r in records])
    axes[1, 0].set_title('Learning rate')
    axes[1, 1].plot(epochs, [r['samples_per_second'] for r in records])
    axes[1, 1].set_title('Cumulative throughput')
    for axis in axes.flat:
        axis.set_xlabel('Completed epochs')
        axis.grid(alpha=0.25)
    fig.suptitle(run_dir.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.png')
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    temporary.replace(output)
    return True


def main():
    args = parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError('refresh-seconds must be positive')
    output = args.output or args.run_dir / 'progress.png'
    while True:
        status = read_status(args.run_dir)
        print(terminal_summary(status), flush=True)
        if not args.no_plot:
            render_plot(args.run_dir, output)
            print(f'plot={output}', flush=True)
        if not args.follow or status['state'] in ('complete', 'failed'):
            break
        time.sleep(args.refresh_seconds)


if __name__ == '__main__':
    main()
