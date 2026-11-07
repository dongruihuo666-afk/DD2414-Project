"""Actual temporary-file deletion/backup tests; never access cluster data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retire_arwen as cleanup


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'source'
        self.root.mkdir()
        (self.root/'checkpoints').mkdir()
        (self.root/'checkpoints/student.pt').write_bytes(b'valuable-checkpoint')
        (self.root/'empty-cache').mkdir()
        self.backup = self.base/'backup'

    def test_full_root_removed_only_after_hash_verified_backup_external_link_target_preserved(self):
        external = self.base/'outside'
        external.write_text('keep')
        (self.root/'shared-link').symlink_to(external)
        result = cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(self.root.exists())
        self.assertEqual(external.read_text(), 'keep')
        self.assertEqual((self.backup/'files/checkpoints/student.pt').read_bytes(), b'valuable-checkpoint')
        self.assertTrue((self.backup/'files/empty-cache').is_dir())
        self.assertTrue((self.backup/'files/shared-link').is_symlink())
        record = json.loads((self.backup/'manifest.json').read_text())
        self.assertEqual(record['status'], 'verified')
        self.assertIn('checkpoints/student.pt', record['sha256_and_entries'])

    def test_inspection_does_not_copy_or_delete(self):
        result = cleanup.retire(self.root, self.backup, lambda:None)
        self.assertEqual(result['status'], 'inspected')
        self.assertTrue(self.root.exists())
        self.assertFalse(self.backup.exists())

    def test_existing_backup_is_never_overwritten(self):
        self.backup.mkdir()
        with self.assertRaises(FileExistsError):
            cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertTrue(self.root.exists())

    def test_bad_copy_blocks_deletion(self):
        original = cleanup.shutil.copytree
        def bad_copy(*args, **kwargs):
            value = original(*args, **kwargs)
            (self.backup/'files/checkpoints/student.pt').write_bytes(b'corrupted')
            return value
        with patch.object(cleanup.shutil, 'copytree', bad_copy), self.assertRaises(RuntimeError):
            cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertEqual((self.root/'checkpoints/student.pt').read_bytes(), b'valuable-checkpoint')

    def test_new_source_file_after_copy_blocks_deletion(self):
        original = cleanup.shutil.copytree
        def changed_source(*args, **kwargs):
            value = original(*args, **kwargs)
            (self.root/'new-file').write_text('preserve')
            return value
        with patch.object(cleanup.shutil, 'copytree', changed_source), self.assertRaises(RuntimeError):
            cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertTrue((self.root/'new-file').exists())

    def test_active_job_check_at_both_boundaries_preserves_source(self):
        with patch.object(cleanup, 'require_idle', side_effect=[None, RuntimeError('other job')]):
            with self.assertRaises(RuntimeError):
                cleanup.retire(self.root, self.backup, lambda:cleanup.require_idle('123'), True)
        self.assertTrue(self.root.exists())
        self.assertTrue((self.backup/'manifest.json').exists())

    def test_symlinked_root_and_special_files_are_rejected(self):
        alias = self.base/'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup.retire(alias, self.backup, lambda:None, True)
        os.mkfifo(self.root/'pipe')
        with self.assertRaises(ValueError):
            cleanup.retire(self.root, self.backup, lambda:None, True)

    def test_changed_owner_is_rejected(self):
        with patch.object(cleanup.os, 'getuid', return_value=os.getuid()+1), self.assertRaises(ValueError):
            cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertTrue(self.root.exists())

    def test_absent_source_and_surprise_dataset_are_handled_without_deletion(self):
        result = cleanup.retire(self.base/'absent', self.backup, lambda:None, True)
        self.assertEqual(result['status'], 'already_absent')
        (self.root/'datasets/nuscenes').mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            cleanup.retire(self.root, self.backup, lambda:None, True)
        self.assertFalse(self.backup.exists())

    def test_wrong_node_cannot_reach_filesystem_cleanup(self):
        with patch.object(cleanup.pwd, 'getpwuid') as user, \
                patch.object(cleanup.socket, 'gethostname', return_value='moria'), \
                patch.dict(os.environ, {'SLURM_JOB_ID':'123', 'SLURM_JOB_NUM_NODES':'1'}), \
                self.assertRaises(ValueError):
            user.return_value.pw_name = 'hongen'
            cleanup.require_context()


if __name__ == '__main__':
    unittest.main()
