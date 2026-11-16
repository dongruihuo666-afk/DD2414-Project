"""Evaluate an untouched completed metric-BEV checkpoint using its original code.

No optimizer, fitted probe, label-driven update, or teacher inference is created.
The archived segmentation decoder is loaded strictly and its random-initial
identity is checked. Outputs are Simple-BEV valid-cell binary segmentation metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

TRAINER_COMMIT = 'a31f9039f9fe3e265a503a1ffdedc03b71e361a0'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(payload, treatment):
    config = payload['run_config']
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    if (payload.get('run_config_sha256') != canonical
            or payload.get('checkpoint_kind') != 'ddp_full_model'
            or config.get('source_code_commit') != TRAINER_COMMIT
            or config.get('training_profile') != 'random-metric-bev'
            or config.get('treatment') != treatment
            or config.get('mode') != 'pretrain'
            or config.get('epochs') != 50
            or config.get('image_resolution') != [448, 800]
            or config.get('nsweeps') != 10
            or config.get('velocity_mode') != 'full'
            or config.get('random_student_initialization', {}).get('kind') != 'random'
            or config.get('official_camera_initialization') is not None
            or config.get('train_samples') != 28130
            or config.get('val_samples') != 6019):
        raise ValueError('Expected the exact completed fifty-epoch pretraining protocol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--treatment', choices=['dino-only', 'dino-metric-depth-bev'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--precision', choices=['float32', 'bf16'], default='float32')
    parser.add_argument('--pilot-only', action='store_true', help='One runtime batch; no score reported')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Positive batch size and nonnegative worker count required')
    source = args.source_root.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != TRAINER_COMMIT:
        raise ValueError('Use the original immutable trainer snapshot')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Original model source must remain clean')
    if args.run_dir.exists():
        raise FileExistsError('Choose a fresh evaluation output directory')
    actual_hash = file_hash(args.checkpoint)
    if actual_hash != args.sha256:
        raise ValueError('Checkpoint SHA-256 does not match the verified archive')
    sys.path[:0] = [str(source / 'scripts'), str(source)]
    import torch
    from baseline_followup import evaluate_predictions, load_datasets, state_digest
    from distributed_training import require_completed_ddp_pretraining
    from fullsize_training import atomic_json_write
    from metric_depth_bev import build_metric_model
    from radar_scaling_data import IndexedSubset, validate_scaling_manifest
    from train_multisweep_teacher import model_inputs
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the complete validation evaluation')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_protocol(payload, args.treatment)
    require_completed_ddp_pretraining(payload)
    config = payload['run_config']
    for field in ('implementation_sha256', 'model_data_implementation_sha256',
                  'additional_implementation_sha256'):
        for name, expected in config[field].items():
            path = source / 'scripts' / name if field == 'implementation_sha256' else source / name
            if file_hash(path) != expected:
                raise ValueError('Original implementation hash mismatch: ' + str(path))
    if file_hash(args.bevcar_source / 'nets/voxelnet.py') != config['bevcar_sha256']:
        raise ValueError('BEVCar source hash differs from pretraining')
    manifest = json.loads((source / 'configs/radar_scaling_manifest_seed125.json').read_text())
    validate_scaling_manifest(manifest)
    if manifest['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Validation manifest differs from pretraining')
    backbone = build_metric_model(device, config['seed'], args.bevcar_source)
    initial_decoder = state_digest(backbone.student.decoder)
    backbone.load_state_dict(payload['model'], strict=True)
    archived_decoder = state_digest(backbone.student.decoder)
    if archived_decoder != initial_decoder:
        raise ValueError('Unexpected trained decoder; do not mislabel the raw-pretraining protocol')
    del payload
    backbone.requires_grad_(False)
    backbone.eval()
    before = state_digest(backbone)
    runtime = SimpleNamespace(data_root=args.data_root, batch_size=args.batch_size,
                              num_workers=args.num_workers, seed=config['seed'])
    datasets = load_datasets(runtime, manifest, 10, resolution=[448, 800])
    for dataset in datasets:
        dataset.data_aug_conf['checkpoint_era_resize'] = True
    args.run_dir.mkdir(parents=True)
    protocol = dict(kind='untouched_pretraining_checkpoint_segmentation', treatment=args.treatment,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=actual_hash,
        original_trainer_commit=revision, decoder_matches_random_initialization=True,
        decoder_sha256=archived_decoder, model_before_sha256=before,
        training_updates=0, fitted_probe=False, pretrained_epochs=50,
        validation_samples=6019, image_resolution=[448, 800], nsweeps=10,
        velocity_mode='full', precision=args.precision, tf32=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        manifest_sha256=manifest['manifest_sha256'],
        metric='global valid-cell binary Vehicle IoU; fixed sigmoid.round threshold',
        interpretation='Archived segmentation decoder is untrained; not supervised Simple-BEV capability')
    atomic_json_write(protocol, args.run_dir / 'config.json')
    started = time.monotonic()

    @torch.inference_mode()
    def predict(batch):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            inputs = model_inputs(batch, device, 'full')
            shared = backbone.features(inputs, 'fused_bev')[0]
            logits = backbone.student.decoder(shared, None)['segmentation']
        if logits.requires_grad or not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite or gradient-tracked evaluation output')
        return logits

    def progress(samples):
        elapsed = time.monotonic() - started
        row = dict(state='evaluating', samples=samples, total_samples=6019,
                   seconds=elapsed, samples_per_second=samples / max(elapsed, 1e-9),
                   updated_unix_time=time.time())
        atomic_json_write(row, args.run_dir / 'status.json')
        print(json.dumps(row), flush=True)

    if args.pilot_only:
        loader = DataLoader(IndexedSubset(datasets[1], manifest['val']['dataset_indices']),
                            batch_size=args.batch_size, num_workers=0)
        _, batch = next(iter(loader))
        logits = predict(batch)
        torch.cuda.synchronize()
        if state_digest(backbone) != before:
            raise RuntimeError('Inference changed parameters or BatchNorm buffers')
        result = dict(state='runtime_pilot_passed', batch_size=logits.shape[0],
                      logit_shape=list(logits.shape), seconds=time.monotonic() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      scope='Runtime admission only; no subset IoU reported')
        atomic_json_write(result, args.run_dir / 'pilot.json')
        print(json.dumps(result), flush=True)
        return
    metrics = evaluate_predictions(predict, datasets[1], manifest['val']['dataset_indices'],
                                   runtime, manifest, args.run_dir / 'validation', progress)
    after = state_digest(backbone)
    if after != before or metrics['samples'] != 6019:
        raise RuntimeError('Model changed or complete validation coverage failed')
    result = dict(status='complete', protocol=protocol, final_validation=metrics,
                  model_after_sha256=after, unchanged_model=True,
                  elapsed_seconds=time.monotonic() - started)
    atomic_json_write(result, args.run_dir / 'summary.json')
    atomic_json_write(dict(state='complete', samples=6019, updated_unix_time=time.time()),
                      args.run_dir / 'status.json')
    print('DIRECT_CHECKPOINT_RESULT ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
