#!/usr/bin/env python3
"""CPU-only tests for the reproducible RPL environment tooling."""

import sys
import tempfile
import unittest
from pathlib import Path


CLUSTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_DIR.parent
sys.path.insert(0, str(CLUSTER_DIR))

import verify_rpl_env as verifier  # noqa: E402


class RplEnvironmentTest(unittest.TestCase):
    def test_sha256_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload'
            path.write_bytes(b'dd2414')
            self.assertEqual(
                verifier.sha256(path),
                'be04fc269c78a98d2fcb270da39c11b327b7da2ff40bd40c69d58d882d6d05c8',
            )

    def test_warning_does_not_hide_required_failure(self):
        audit = verifier.Audit()
        audit.warn('optional', 'not present')
        self.assertFalse(audit.failures)
        audit.fail('required', 'broken')
        self.assertEqual([item['name'] for item in audit.failures], ['required'])

    def test_cpu_only_cuda_check_fails_in_gpu_mode(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        audit = verifier.Audit()
        verifier.check_gpu(audit, FakeTorch(), 'fp16')
        self.assertEqual([item['name'] for item in audit.failures],
                         ['cuda_available'])
        self.assertIn('do not count', audit.failures[0]['detail'])

    def test_requirements_match_the_verified_direct_pins(self):
        pins = {}
        requirements = CLUSTER_DIR / 'requirements-rpl.txt'
        for raw_line in requirements.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            name, version = line.split('==', 1)
            pins[name] = version
        expected = {
            name: version for name, version in verifier.EXPECTED_PACKAGES.items()
            if name not in {'torch', 'torchvision'}
        }
        self.assertEqual(pins, expected)

    def test_launchers_expose_required_safety_controls(self):
        train = (CLUSTER_DIR / 'train.sh').read_text()
        smoke = train.split('run_smoke() {', 1)[1].split(
            'verify_gpu_environment() {', 1,
        )[0]
        self.assertEqual(smoke.count('--precision "${PRECISION}"'), 2)
        submit = (CLUSTER_DIR / 'submit_slurm.sh').read_text()
        self.assertIn('SLURM_CONSTRAINT', submit)
        self.assertIn('limited to exactly one GPU', submit)
        self.assertIn('may not exceed the 48-hour', submit)


if __name__ == '__main__':
    unittest.main()
