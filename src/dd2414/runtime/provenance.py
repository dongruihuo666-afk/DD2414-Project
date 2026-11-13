"""Content verification and read-only source provenance; never publishes Git state."""
import hashlib
import subprocess
from pathlib import Path
from dd2414.paths import repository_root
ROOT = repository_root()

def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
    ).strip()

def record_package_implementation(config, root):
    """Extend existing provenance fields with the actual installed package code.

    Training requires matching source-checkout bytes. This does not translate
    old source hashes, change the checkpoint schema or relax exact resume.
    """
    package = Path(__file__).resolve().parents[1]
    records = dict(config.get('model_data_implementation_sha256', {}))
    for path in sorted(package.rglob('*.py')):
        relative = Path('src/dd2414') / path.relative_to(package)
        expected = Path(root) / relative
        actual_hash = digest_file(path)
        if not expected.is_file() or digest_file(expected) != actual_hash:
            raise ValueError('Installed package differs from training checkout: ' + str(relative))
        records[relative.as_posix()] = actual_hash
    config['model_data_implementation_sha256'] = records
    return config


def check_camera_source_provenance(config, checkpoint_path, initialization, allow_relocation=False):
    expected = str(checkpoint_path.resolve()) if checkpoint_path else None
    original = config.get('official_camera_checkpoint')
    stored = config.get('official_camera_initialization')
    if not allow_relocation:
        if original != expected:
            raise ValueError('source official encoder provenance mismatch')
        if initialization and stored != initialization:
            raise ValueError('source official encoder hash mismatch')
        return
    if not checkpoint_path or not initialization or not stored:
        raise ValueError('Relocation requires both explicit official encoder records')
    # Only the location changes. Hash, tensor count, and encoder-only scope must
    # still agree, and model.load_state_dict(strict=True) remains unchanged.
    if {k:v for k,v in stored.items() if k != 'path'} != {
            k:v for k,v in initialization.items() if k != 'path'}:
        raise ValueError('source official encoder content/provenance mismatch')


def check_continuation_source(source, expected, manifest_sha256, checkpoint_path,
                              initialization, allow_relocation=False):
    """Validate a completed no-depth parent for a NEW, equal-budget phase.

    This deliberately does not restore optimizer/sampler/RNG state. Exact
    interrupted-run resume remains the responsibility of training_checkpoint.
    """
    cfg = source['run_config']
    if (cfg.get('experiment') != 'baseline_multisweep_double_distillation'
            or cfg.get('mode') != 'pretrain' or cfg.get('depth_mode') != 'none'
            or cfg.get('epochs') != 5 or source['sampler']['epoch'] != 5
            or source['progress']['epoch'] != 5
            or source['progress']['update'] != 140650
            or source['progress'].get('validations', {}).get('5', {}).get('samples') != 6019):
        raise ValueError('Continuation requires the completed five-epoch no-depth pretraining model')
    if set(expected) != {'velocity_mode'} or expected['velocity_mode'] not in ['full','zero']:
        raise ValueError('Only the matching velocity treatment may be selected for continuation')
    required = dict(nsweeps=10, seed=125, image_resolution=[448,800],
                    teacher_resolution=[448,798], batch_size=1,
                    train_samples=28130, val_samples=6019,
                    image_weight=.2, bev_weight=.8, motion_weight=.5,
                    precision='bf16')
    required.update(expected)
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'Continuation source {key} mismatch')
    if cfg.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Continuation source manifest mismatch')
    check_camera_source_provenance(cfg, checkpoint_path, initialization, allow_relocation)
    return dict(source_epochs=cfg['epochs'], source_velocity_mode=cfg['velocity_mode'],
                source_code_commit=cfg.get('source_code_commit'),
                optimizer_restored=False, sampler_restarted=True,
                scope='All model tensors loaded strictly; NEW optimizer/sampler/RNG schedule, not exact resume')
