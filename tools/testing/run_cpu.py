"""Run CPU regressions, excluding dataset and visualization experiments."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEST_DIRECTORIES = ("scripts", "cluster_rpl", "cluster_experiment")
SUITES = (
    'test_fullsize_training',
    'test_baseline_followup',
    'test_multisweep_teacher',
    'test_metric_depth_bev',
    'test_metric_memory',
    'test_depth_objective',
    'test_depth_continuation',
    'test_distributed_training',
    'test_training_checkpoint',
    'test_radar_evaluation',
    'test_full_data_protocol',
    'test_radar_scaling_data',
    'test_fullsize_linear_probe',
    'test_trainval_configuration',
    'test_official_full_data',
    'test_continue_recovered_depth',
    'test_depth_assets',
    'test_depth_study',
    'test_depth_pilot',
    'test_random_depth_pilot',
    'test_random_depth_study',
    'test_random_depth_migration',
    'test_retire_arwen',
    'test_recovery_checkpoint',
    'test_cluster',
    'test_analyze_radar_scaling_curve',
    'test_hybrid_teacher_protocol',
    'test_rpl_environment',
)


def main():
    os.chdir(ROOT)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR="/tmp/dd2414-independent-mpl",
    )
    sys.path[:0] = [
        str(ROOT / "src"), str(ROOT),
        *(str(ROOT / directory) for directory in TEST_DIRECTORIES),
    ]
    available = [
        name for name in SUITES
        if any((ROOT / directory / (name + ".py")).is_file()
               for directory in TEST_DIRECTORIES)
    ]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in available
    )
    unit_tests = ROOT / "tests/unit"
    if unit_tests.is_dir():
        suite.addTests(unittest.defaultTestLoader.discover(str(unit_tests)))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
