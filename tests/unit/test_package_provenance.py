"""The formal package must be covered by strict source provenance."""
import shutil
import tempfile
from pathlib import Path
import unittest

from dd2414.paths import repository_root
from dd2414.runtime.provenance import record_package_implementation


class PackageProvenanceTests(unittest.TestCase):
    def test_existing_source_fields_preserved_and_actual_package_files_recorded(self):
        config={'model_data_implementation_sha256':{'existing.py':'existing-digest'}}
        result=record_package_implementation(config,repository_root(required=True))
        self.assertIs(result,config)
        self.assertEqual(result['model_data_implementation_sha256']['existing.py'],'existing-digest')
        self.assertIn('src/dd2414/models/metric_bev.py',result['model_data_implementation_sha256'])
        self.assertIn('src/dd2414/runtime/checkpoints.py',result['model_data_implementation_sha256'])

    def test_mismatched_installed_and_checkout_source_is_rejected(self):
        root=repository_root(required=True)
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/'src/dd2414'
            shutil.copytree(root/'src/dd2414',copy,ignore=shutil.ignore_patterns('__pycache__'))
            file=copy/'losses/semantic.py';file.write_text(file.read_text()+'\n# mismatched checkout\n')
            with self.assertRaisesRegex(ValueError,'Installed package differs'):
                record_package_implementation({},Path(directory))

    def test_missing_checkout_source_cannot_be_recorded_as_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):record_package_implementation({},Path(directory))


if __name__=='__main__':unittest.main()
