#!/usr/bin/env python3
"""Deterministic nested manifests and streaming loaders for radar scaling."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SEED = 125
DEFAULT_SCALE_COUNTS = (64, 256, 1024, 4096)


class IndexedSubset(Dataset):
    """Subset that also returns its stable position for token-level logging."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return position, self.dataset[self.indices[position]]


def _stable_rank(seed, namespace, token):
    value = f'{seed}:{namespace}:{token}'.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def dataset_records(dataset):
    """Describe dataset positions without loading any sensor payloads."""
    records = []
    for dataset_index, sequence in enumerate(dataset.indices):
        if len(sequence) != 1:
            raise ValueError('radar scaling currently requires seqlen=1')
        sample_index = int(sequence[0])
        row = dataset.ixes[sample_index]
        records.append({
            'dataset_index': dataset_index,
            'sample_index': sample_index,
            'token': row['token'],
            'scene_token': row['scene_token'],
        })
    tokens = [record['token'] for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError('dataset contains duplicate sample tokens')
    return records


def scene_spread_order(records, seed=DEFAULT_SEED):
    """Return a deterministic order that round-robins across scenes.

    Scene and within-scene orders use SHA-256 instead of Python's randomized
    ``hash`` so manifests remain identical across machines and processes.
    """
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record['scene_token']].append(record)
    scenes = sorted(
        by_scene,
        key=lambda token: (_stable_rank(seed, 'scene', token), token),
    )
    for scene_token, scene_records in by_scene.items():
        scene_records.sort(
            key=lambda record: (
                _stable_rank(seed, f'sample:{scene_token}', record['token']),
                record['token'],
            )
        )
    ordered = []
    depth = 0
    while len(ordered) < len(records):
        added = 0
        for scene_token in scenes:
            scene_records = by_scene[scene_token]
            if depth < len(scene_records):
                ordered.append(scene_records[depth])
                added += 1
        if added == 0:
            raise RuntimeError('scene round-robin stopped before covering all records')
        depth += 1
    return ordered


def _record_columns(records):
    return {
        'dataset_indices': [record['dataset_index'] for record in records],
        'sample_indices': [record['sample_index'] for record in records],
        'tokens': [record['token'] for record in records],
        'scene_tokens': [record['scene_token'] for record in records],
        'unique_scenes': len({record['scene_token'] for record in records}),
    }


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def build_scaling_manifest(train_dataset, val_dataset, seed=DEFAULT_SEED,
                           scale_counts=DEFAULT_SCALE_COUNTS):
    train_records = dataset_records(train_dataset)
    val_records = dataset_records(val_dataset)
    train_scenes = {record['scene_token'] for record in train_records}
    val_scenes = {record['scene_token'] for record in val_records}
    overlap = train_scenes & val_scenes
    if overlap:
        raise ValueError(f'train/val scene overlap: {len(overlap)} scenes')

    ordered_train = scene_spread_order(train_records, seed)
    counts = tuple(int(count) for count in scale_counts)
    if tuple(sorted(set(counts))) != counts:
        raise ValueError('scale counts must be unique and strictly increasing')
    if counts and (counts[0] < 1 or counts[-1] >= len(ordered_train)):
        raise ValueError('bounded scales must be positive and smaller than full train')

    scales = {}
    for count in counts:
        selected = ordered_train[:count]
        scales[str(count)] = {'count': count, **_record_columns(selected)}
    scales['full'] = {'count': len(ordered_train), **_record_columns(ordered_train)}

    val = {'count': len(val_records), **_record_columns(val_records)}
    manifest = {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'seed': int(seed),
        'dataset_version': getattr(getattr(train_dataset, 'nusc', None),
                                   'version', 'unknown'),
        'selection': 'stable-hash scene round-robin; each scale is a prefix',
        'scales': scales,
        'val': val,
    }
    manifest['manifest_sha256'] = _digest(manifest)
    validate_scaling_manifest(manifest)
    return manifest


def validate_scaling_manifest(manifest):
    if manifest.get('schema_version') != MANIFEST_SCHEMA_VERSION:
        raise ValueError('unsupported manifest schema version')
    scales = manifest['scales']
    full = scales['full']
    full_indices = full['dataset_indices']
    full_tokens = full['tokens']
    if full['count'] != len(full_indices) or full['count'] != len(full_tokens):
        raise ValueError('full scale count does not match its records')
    if len(full_indices) != len(set(full_indices)):
        raise ValueError('full scale repeats dataset indices')
    if len(full_tokens) != len(set(full_tokens)):
        raise ValueError('full scale repeats sample tokens')

    bounded = sorted(
        ((int(name), value) for name, value in scales.items() if name != 'full'),
        key=lambda item: item[0],
    )
    previous_indices = []
    for count, value in bounded:
        if value['count'] != count:
            raise ValueError(f'scale {count} stores the wrong count')
        indices = value['dataset_indices']
        if indices != full_indices[:count]:
            raise ValueError(f'scale {count} is not a prefix of full')
        if indices[:len(previous_indices)] != previous_indices:
            raise ValueError(f'scale {count} is not nested')
        if len(value['tokens']) != count or len(value['scene_tokens']) != count:
            raise ValueError(f'scale {count} has incomplete record columns')
        previous_indices = indices

    val = manifest['val']
    if val['count'] != len(val['dataset_indices']) or val['count'] != len(val['tokens']):
        raise ValueError('validation count does not match its records')
    if set(full['scene_tokens']) & set(val['scene_tokens']):
        raise ValueError('manifest train/val scenes overlap')

    expected_digest = manifest.get('manifest_sha256')
    if expected_digest is not None:
        unsigned = dict(manifest)
        del unsigned['manifest_sha256']
        if _digest(unsigned) != expected_digest:
            raise ValueError('manifest digest does not match its content')
    return True


def write_manifest(manifest, path):
    """Atomically write a validated JSON manifest."""
    validate_scaling_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, path)


def make_streaming_loader(dataset, dataset_indices, batch_size=1, num_workers=0):
    """Create a loader that fetches on demand and never materializes batches."""
    indices = list(dataset_indices)
    if not indices:
        raise ValueError('streaming loader requires at least one dataset index')
    if len(indices) != len(set(indices)):
        raise ValueError('streaming loader indices must be unique')
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError('streaming loader index outside dataset')
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )


def scale_indices(manifest, scale):
    key = str(scale)
    if key not in manifest['scales']:
        raise KeyError(f'unknown scale {scale}; choose from {list(manifest["scales"])}')
    return manifest['scales'][key]['dataset_indices']


def prefix_indices(manifest, count):
    count = int(count)
    full = manifest['scales']['full']['dataset_indices']
    if not 1 <= count <= len(full):
        raise ValueError('prefix count must fit the full training scale')
    return full[:count]
