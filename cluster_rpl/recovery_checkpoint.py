"""Audited loader-only checkpoint relocation; never resets learned/resume state."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def resume_digest(payload):
    """Hash ALL non-config state, including optimizer, cursors, rank RNG and BN."""
    digest = hashlib.sha256()
    def visit(value):
        digest.update(type(value).__name__.encode()+b'\0')
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()+repr(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode()+b'\0')
    visit({k:v for k,v in payload.items() if k not in ('run_config', 'run_config_sha256')})
    return digest.hexdigest()


def migrate(source, destination, training_root, expected_update, treatment):
    """Only num_workers 2->0 and the output path may differ; source stays intact."""
    source, destination, training_root = map(Path, (source, destination, training_root))
    if source.is_symlink() or destination.exists():
        raise ValueError('Use an existing real parent checkpoint and a fresh output')
    parent_hash = file_digest(source)
    value = torch.load(source, map_location='cpu', weights_only=False)
    config, progress = value['run_config'], value['progress']
    if (value.get('schema_version') != 2 or value.get('checkpoint_kind') != 'ddp_full_model'
            or value.get('world_size') != 5 or config.get('num_workers') != 2
            or config.get('training_profile') != 'random-metric-bev' or config.get('treatment') != treatment
            or config.get('epochs') != 50 or config.get('train_samples') != 28130
            or config.get('val_samples') != 6019 or progress['update'] != expected_update
            or progress['samples_seen'] != expected_update*5
            or value['run_config_sha256'] != canonical(config)):
        raise ValueError('Unexpected parent provenance, budget, config integrity or cursor')
    states = value['distributed_state']
    if len(states) != 5 or any(state['rank'] != rank or state['sampler']['position'] != value['sampler']['position']
            or state['sampler']['epoch'] != value['sampler']['epoch'] for rank,state in enumerate(states)):
        raise ValueError('Incomplete/inconsistent per-rank state')
    tensors = list(value['model'].values()) + [x for row in value['optimizer']['state'].values()
                                               for x in row.values() if isinstance(x, torch.Tensor)]
    if not all(not torch.is_floating_point(x) or bool(torch.isfinite(x).all()) for x in tensors):
        raise ValueError('Parent learned/optimizer state is non-finite')
    for field in ('implementation_sha256', 'model_data_implementation_sha256', 'additional_implementation_sha256'):
        for name, expected in config[field].items():
            path = training_root/'scripts'/name if field == 'implementation_sha256' else training_root/name
            if file_digest(path) != expected:
                raise ValueError('Training implementation changed: '+str(path))
    before = resume_digest(value)
    updated = copy.deepcopy(config)
    updated.update(num_workers=0, run_dir=str(destination.resolve()))
    allowed = {key for key in config if config[key] != updated[key]}
    if allowed != {'num_workers', 'run_dir'}:
        raise ValueError('Runtime-only migration must change exactly workers/output path')
    value['run_config'], value['run_config_sha256'] = updated, canonical(updated)
    destination.mkdir(parents=True)
    checkpoint = destination/'checkpoints/latest.pt'
    checkpoint.parent.mkdir()
    temporary = checkpoint.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(checkpoint)
    recovered = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if resume_digest(recovered) != before or file_digest(source) != parent_hash:
        raise RuntimeError('Learned/resume state changed or parent was modified')
    (destination/'config.json').write_text(json.dumps(updated, indent=2)+'\n')
    report = dict(kind='audited-runtime-only-resume', source=str(source), source_sha256=parent_hash,
        destination=str(checkpoint), migrated_checkpoint_sha256=file_digest(checkpoint),
        source_config_sha256=canonical(config), destination_config_sha256=canonical(updated),
        config_changes={key:dict(before=config[key],after=updated[key]) for key in sorted(allowed)},
        unchanged_resume_state_sha256=before, update=expected_update, samples_seen=progress['samples_seen'],
        scope='Exact model/optimizer/progress/rank RNG/BN/sampler preservation; loader count and fresh output only')
    (destination/'runtime_migration.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
