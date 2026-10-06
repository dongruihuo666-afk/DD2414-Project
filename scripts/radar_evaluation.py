#!/usr/bin/env python3
"""Deterministic radar controls and incremental per-frame evaluation records."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch


RADAR_MODES = ('matched', 'zero_velocity', 'empty', 'wrong_scene')
BEVCAR_VELOCITY_CHANNELS = slice(4, 6)


def _stable_rank(seed, token):
    return hashlib.sha256(f'{seed}:wrong-scene:{token}'.encode('utf-8')).hexdigest()


def build_wrong_scene_map(tokens, scene_tokens, seed=125):
    """Build a deterministic bijection whose source scene always differs.

    Samples are grouped into stable-hash-ordered scene blocks, then rotated by
    the largest scene size. If no scene owns more than half the samples, this
    guarantees that a block cannot overlap itself after rotation. The mapping
    preserves the exact validation radar-token multiset instead of over-sampling
    a few source scenes.
    """
    if len(tokens) != len(scene_tokens) or len(tokens) < 2:
        raise ValueError('wrong-scene mapping requires at least two aligned samples')
    if len(tokens) != len(set(tokens)):
        raise ValueError('wrong-scene mapping requires unique sample tokens')
    if len(set(scene_tokens)) < 2:
        raise ValueError('wrong-scene mapping requires at least two scenes')
    by_scene = defaultdict(list)
    for token, scene_token in zip(tokens, scene_tokens):
        by_scene[scene_token].append(token)
    scenes = sorted(
        by_scene,
        key=lambda scene: (_stable_rank(seed, scene), scene),
    )
    records = []
    for scene in scenes:
        for token in sorted(
                by_scene[scene], key=lambda value: (_stable_rank(seed, value), value)):
            records.append((token, scene))
    count = len(records)
    shift = max(len(values) for values in by_scene.values())
    if shift * 2 > count:
        raise ValueError('no scene-level derangement exists: one scene owns over half')
    mapping = {
        records[index][0]: records[(index + shift) % count][0]
        for index in range(count)
    }
    validate_wrong_scene_map(mapping, tokens, scene_tokens)
    return mapping


def validate_wrong_scene_map(mapping, tokens, scene_tokens):
    scenes = dict(zip(tokens, scene_tokens))
    if set(mapping) != set(tokens):
        raise ValueError('wrong-scene mapping does not cover every target token')
    if set(mapping.values()) != set(tokens):
        raise ValueError('wrong-scene mapping is not a source-token bijection')
    for target, source in mapping.items():
        if scenes[target] == scenes[source]:
            raise ValueError(f'wrong-scene pair shares a scene: {target} -> {source}')
    return True


def wrong_scene_dataset_indices(manifest, seed=125):
    val = manifest['val']
    mapping = build_wrong_scene_map(val['tokens'], val['scene_tokens'], seed=seed)
    index_by_token = dict(zip(val['tokens'], val['dataset_indices']))
    return [index_by_token[mapping[token]] for token in val['tokens']], mapping


def radar_for_mode(matched_voxels, mode, wrong_scene_voxels=None):
    """Return BEVCar voxel input for one of the locked evaluation controls."""
    features, coords, counts = matched_voxels
    if mode == 'matched':
        return matched_voxels
    if mode == 'wrong_scene':
        if wrong_scene_voxels is None:
            raise ValueError('wrong_scene mode requires wrong-scene voxels')
        return wrong_scene_voxels
    if mode == 'zero_velocity':
        output = features.clone()
        output[..., BEVCAR_VELOCITY_CHANNELS] = 0.0
        return output, coords, counts
    if mode == 'empty':
        return torch.zeros_like(features), torch.zeros_like(coords), torch.zeros_like(counts)
    raise ValueError(f'unknown radar mode: {mode}')


class IncrementalEvaluationWriter:
    """Write one compact JSON record per frame without retaining prior results."""

    def __init__(self, path, modes=RADAR_MODES):
        self.path = Path(path)
        self.partial_path = self.path.with_name(f'.{self.path.name}.partial')
        self.modes = tuple(modes)
        self.handle = None
        self.tokens = set()
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.partial_path.open('w')
        return self

    def write(self, record):
        if self.handle is None:
            raise RuntimeError('evaluation writer is not open')
        token = record['token']
        if token in self.tokens:
            raise ValueError(f'duplicate evaluation token: {token}')
        if tuple(record['modes']) != self.modes:
            raise ValueError(f'evaluation modes must be exactly {self.modes}')
        if record['scene_token'] == record['wrong_radar_scene_token']:
            raise ValueError('wrong radar must come from a different scene')
        self.handle.write(json.dumps(
            record, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ) + '\n')
        self.handle.flush()
        self.tokens.add(token)
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            if exc_type is None:
                self.handle.flush()
                os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None
        if exc_type is None:
            os.replace(self.partial_path, self.path)
        return False
