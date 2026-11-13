import hashlib
import subprocess
from pathlib import Path
from dd2414.paths import repository_root
ROOT=repository_root()
def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
