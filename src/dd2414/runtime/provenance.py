import hashlib
import subprocess
from pathlib import Path
from dd2414.paths import repository_root
ROOT=repository_root()
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
