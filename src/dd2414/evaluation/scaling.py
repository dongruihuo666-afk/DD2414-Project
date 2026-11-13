"""Preserved streaming validation and log helpers for the scaling study."""
import hashlib
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
import torch
from dd2414.models.legacy_motion import prepare_bevcar_inputs
from dd2414.objectives.radar_motion import motion_target, motion_loss
from dd2414.teachers.dino import build_targets
from dd2414.losses.semantic import semantic_loss
from dd2414.data.manifest import make_streaming_loader
from dd2414.evaluation.radar_controls import (RADAR_MODES, spread_positions,
    wrong_scene_dataset_indices, IncrementalEvaluationWriter, radar_for_mode,
    apply_radar_velocity_mode)

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_train_log(path, resume_update=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume_update is None:
        if path.exists():
            raise FileExistsError(f'train log already exists: {path}')
        return
    if not path.exists():
        raise FileNotFoundError(f'resume train log is missing: {path}')
    retained = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if int(record['update']) <= int(resume_update):
            retained.append(record)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(''.join(
        json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        for record in retained
    ))
    os.replace(temporary, path)


def append_train_record(path, record):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(
            record, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ) + '\n')
        handle.flush()


def trainable_optimizer(model, learning_rate):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    return parameters, optimizer


def _empty_metric_totals():
    return {mode: {'semantic': 0.0, 'motion': 0.0,
                   'moving_motion': 0.0, 'cell_motion': 0.0}
            for mode in RADAR_MODES}


def _metric_means(totals, count, moving_frames, covered_cells):
    means = {}
    for mode, values in totals.items():
        means[mode] = {
            'semantic_loss': values['semantic'] / count,
            'motion_loss': values['motion'] / count,
            'moving_frame_motion_loss': (
                values['moving_motion'] / moving_frames if moving_frames else None
            ),
            'cell_weighted_motion_loss': (
                values['cell_motion'] / covered_cells if covered_cells else None
            ),
        }
    return means


def _paired_penalties(means):
    penalties = {}
    for metric in ('semantic_loss', 'motion_loss',
                   'moving_frame_motion_loss', 'cell_weighted_motion_loss'):
        matched = means['matched'][metric]
        penalties[metric] = {}
        for mode in ('zero_velocity', 'empty', 'wrong_scene'):
            value = means[mode][metric]
            penalties[metric][mode] = (
                value - matched if value is not None and matched is not None else None
            )
    return penalties


def validation_summary(model, teacher, dataset, manifest, count, device, seed,
                       output_path, use_amp, print_every=500,
                       radar_velocity_mode='full'):
    val = manifest['val']
    if not 1 <= count <= val['count']:
        raise ValueError('val-samples must fit the full validation split')
    positions = spread_positions(val['count'], count)
    wrong_indices, wrong_map = wrong_scene_dataset_indices(manifest, seed=seed)
    target_indices = [val['dataset_indices'][position] for position in positions]
    source_indices = [wrong_indices[position] for position in positions]
    targets = make_streaming_loader(dataset, target_indices, num_workers=0)
    sources = make_streaming_loader(dataset, source_indices, num_workers=0)
    scene_by_token = dict(zip(val['tokens'], val['scene_tokens']))
    totals = _empty_metric_totals()
    scene_totals = defaultdict(_empty_metric_totals)
    scene_counts = defaultdict(int)
    scene_moving_frames = defaultdict(int)
    scene_covered_cells = defaultdict(int)
    moving_frames = 0
    covered_cells = 0
    model.eval()
    start = time.perf_counter()
    with IncrementalEvaluationWriter(output_path) as writer, torch.inference_mode():
        for offset, (batch, wrong_batch) in enumerate(zip(targets, sources)):
            fusion_targets, confidences, _ = build_targets([batch], teacher, device)
            inputs = prepare_bevcar_inputs(batch, device)
            wrong_inputs = prepare_bevcar_inputs(wrong_batch, device)
            motion_field, motion_coverage, motion_info = motion_target(batch, device)
            images, pixels, cameras, vox_util, matched_voxels = inputs
            cells = motion_info['covered_cells']
            moving_frames += int(cells > 0)
            covered_cells += cells
            modes = {}
            for mode in RADAR_MODES:
                radar = radar_for_mode(
                    matched_voxels, mode, wrong_scene_voxels=wrong_inputs[-1],
                )
                radar = apply_radar_velocity_mode(radar, radar_velocity_mode)
                amp = torch.autocast(device_type='cuda', dtype=torch.float16) \
                    if use_amp else nullcontext()
                with amp:
                    semantic_prediction, motion_prediction = model(
                        images, pixels, cameras, vox_util, radar,
                    )
                    sem, _ = semantic_loss(
                        semantic_prediction, fusion_targets[0][None],
                        confidences[0][None],
                    )
                    mot, _ = motion_loss(
                        motion_prediction, motion_field, motion_coverage,
                    )
                if not torch.isfinite(sem) or not torch.isfinite(mot):
                    raise RuntimeError(
                        f'non-finite validation loss at position {positions[offset]} mode {mode}'
                    )
                sem_value, mot_value = float(sem), float(mot)
                modes[mode] = {'semantic_loss': sem_value, 'motion_loss': mot_value}
                totals[mode]['semantic'] += sem_value
                totals[mode]['motion'] += mot_value
                if cells > 0:
                    totals[mode]['moving_motion'] += mot_value
                    totals[mode]['cell_motion'] += mot_value * cells
            manifest_position = positions[offset]
            token = val['tokens'][manifest_position]
            scene_token = val['scene_tokens'][manifest_position]
            wrong_token = wrong_map[token]
            scene_counts[scene_token] += 1
            scene_moving_frames[scene_token] += int(cells > 0)
            scene_covered_cells[scene_token] += cells
            for mode in RADAR_MODES:
                scene_totals[scene_token][mode]['semantic'] += modes[mode]['semantic_loss']
                scene_totals[scene_token][mode]['motion'] += modes[mode]['motion_loss']
                if cells > 0:
                    scene_totals[scene_token][mode]['moving_motion'] += modes[mode]['motion_loss']
                    scene_totals[scene_token][mode]['cell_motion'] += modes[mode]['motion_loss'] * cells
            writer.write({
                'token': token,
                'scene_token': scene_token,
                'wrong_radar_token': wrong_token,
                'wrong_radar_scene_token': scene_by_token[wrong_token],
                'motion_covered_cells': cells,
                'modes': modes,
            })
            del fusion_targets, confidences, inputs, wrong_inputs
            if (offset + 1) % print_every == 0 or offset + 1 == count:
                print(f'validation={offset + 1}/{count}', flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    means = _metric_means(totals, count, moving_frames, covered_cells)
    per_scene = {}
    for scene_token in sorted(scene_counts):
        scene_means = _metric_means(
            scene_totals[scene_token], scene_counts[scene_token],
            scene_moving_frames[scene_token], scene_covered_cells[scene_token],
        )
        per_scene[scene_token] = {
            'samples': scene_counts[scene_token],
            'moving_frames': scene_moving_frames[scene_token],
            'covered_cells': scene_covered_cells[scene_token],
            'means': scene_means,
            'penalties': _paired_penalties(scene_means),
        }
    return {
        'radar_velocity_mode': radar_velocity_mode,
        'samples': count,
        'moving_frames': moving_frames,
        'covered_cells': covered_cells,
        'positions': positions,
        'seconds': elapsed,
        'frames_per_second': count / elapsed,
        'amp': use_amp,
        'means': means,
        'penalties': _paired_penalties(means),
        'per_scene': per_scene,
        'records': str(output_path),
    }
