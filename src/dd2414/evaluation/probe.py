"""Shared probe metrics and deterministic initialization, independent of entrypoints."""
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def balanced_binary_loss(logits, target, valid):
    """Average foreground and background BCE without empty-class NaNs."""
    raw = F.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                              reduction='none')
    valid_mask = valid.gt(0.5)
    terms = []
    for class_mask in (target.gt(0.5), target.le(0.5)):
        mask = valid_mask & class_mask
        if mask.any():
            terms.append(raw[mask].mean())
    if not terms:
        raise ValueError('probe batch has no valid BEV cells')
    return torch.stack(terms).mean()


def binary_segmentation_counts(logits, target, valid):
    prediction = logits.sigmoid().ge(0.5)
    truth = target.gt(0.5)
    mask = valid.gt(0.5)
    return {
        'tp': int((prediction & truth & mask).sum()),
        'fp': int((prediction & ~truth & mask).sum()),
        'fn': int((~prediction & truth & mask).sum()),
        'tn': int((~prediction & ~truth & mask).sum()),
    }


def metrics_from_counts(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        'vehicle_iou': tp / iou_denominator if iou_denominator else None,
        'precision': tp / precision_denominator if precision_denominator else None,
        'recall': tp / recall_denominator if recall_denominator else None,
        'f1': 2 * tp / f1_denominator if f1_denominator else None,
        **counts,
    }


def add_counts(total, current):
    for key in total:
        total[key] += current[key]


def checkpoint_velocity_mode(payload):
    return payload['run_config'].get('radar_velocity_mode', 'full')


def resolve_velocity_mode(requested, payload):
    stored = checkpoint_velocity_mode(payload)
    if requested == 'auto':
        return stored
    if requested != stored:
        raise ValueError(
            f'probe velocity mode {requested} does not match checkpoint mode {stored}'
        )
    return requested
