"""Fail-closed CPU admission contracts; no dataset, GPU or Slurm allocation."""
import unittest
from check_depth_assets import check_sensor_coverage, require_cpu_allocation


class DepthAssetTests(unittest.TestCase):
    def test_audit_refuses_login_and_wrong_data_node(self):
        config = dict(node='arwen')
        require_cpu_allocation(config, dict(SLURM_JOB_ID='1'), 'arwen')
        for environment, host in [({}, 'arwen'), (dict(SLURM_JOB_ID='1'), 'moria')]:
            with self.assertRaises(ValueError):
                require_cpu_allocation(config, environment, host)

    def test_audit_requires_every_referenced_sensor_file(self):
        complete = dict(referenced=2631083, missing=0, empty=0, unsafe=0)
        check_sensor_coverage(complete)
        for key, value in [('referenced', 2631082), ('missing', 1), ('empty', 1), ('unsafe', 1)]:
            with self.assertRaises(ValueError):
                check_sensor_coverage(dict(complete, **{key: value}))


if __name__ == '__main__':
    unittest.main()
