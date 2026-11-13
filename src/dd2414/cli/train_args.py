"""Unchanged research CLI flags, importable without Torch or external assets."""
import argparse
from pathlib import Path

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--bevcar-source', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--mode', choices=['pretrain','frozen-probe','unfrozen-probe',
                                    'finetune','object-motion'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--pretrain-initial-checkpoint', type=Path,
                   help='Opt-in NEW pretraining phase from a completed no-depth model; not resume.')
    p.add_argument('--pretrain-initial-sha256',
                   help='Required pinned SHA-256 for the pretraining-phase parent.')
    p.add_argument('--official-camera-checkpoint', type=Path)
    p.add_argument('--allow-source-relocation', action='store_true',
                   help='Allow a downstream parent to reference an identical official RGB file at a new path; exact resume stays strict.')
    p.add_argument('--image-resolution', type=int, nargs=2, default=[112,192])
    p.add_argument('--teacher-resolution', type=int, nargs=2, default=[196,336])
    p.add_argument('--depth-teacher-resolution', type=int, nargs=2, default=[196,336])
    p.add_argument('--velocity-mode', choices=['full','zero'], default='full')
    p.add_argument('--probe-location', choices=['camera_bev','fused_bev'], default='fused_bev')
    p.add_argument('--initial-camera-probe', action='store_true')
    p.add_argument('--nsweeps', type=int, choices=[5,10], default=5)
    p.add_argument('--depth-mode', choices=['none','radar','radar-dav2','metric-bev'], default='none')
    p.add_argument('--training-profile', choices=['legacy-continuation','random-metric-bev'], default='legacy-continuation')
    p.add_argument('--treatment', choices=['dino-only','dino-metric-depth-bev'])
    p.add_argument('--metric-teacher-source', type=Path)
    p.add_argument('--metric-teacher-weights', type=Path)
    p.add_argument('--metric-teacher-sha256')
    p.add_argument('--metric-depth-weight', type=float, default=.05)
    p.add_argument('--metric-depth-sigma', type=float, default=2.)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=125)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--camera-lr', type=float, default=1e-5)
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--image-weight', type=float, default=.2)
    p.add_argument('--bev-weight', type=float, default=.8)
    p.add_argument('--motion-weight', type=float, default=.5)
    p.add_argument('--depth-weight', type=float, default=.2)
    p.add_argument('--relative-depth-weight', type=float, default=.05)
    p.add_argument('--precision', choices=['bf16','fp16'], default='bf16')
    p.add_argument('--train-samples', type=int, choices=(0,28130), default=0,
                   help='Full split only; subset training has been removed.')
    p.add_argument('--val-samples', type=int, choices=(6019,), default=6019)
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--print-every', type=int, default=25)
    p.add_argument('--stop-after-updates', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--distributed', action='store_true',
                   help='Opt-in two-rank full-data pretraining continuation; downstream remains single-GPU.')
    return p.parse_args(argv)
