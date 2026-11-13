#!/usr/bin/env python3
"""Explicit CPU regression suite; does not discover dataset/plotting experiments."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    'test_fullsize_training', 'test_baseline_followup', 'test_multisweep_teacher',
    'test_metric_depth_bev', 'test_metric_memory', 'test_depth_objective',
    'test_depth_continuation', 'test_distributed_training', 'test_training_checkpoint',
    'test_radar_evaluation', 'test_full_data_protocol', 'test_radar_scaling_data',
    'test_fullsize_linear_probe', 'test_trainval_configuration',
    'test_official_full_data', 'test_continue_recovered_depth',
    'test_depth_assets', 'test_depth_study', 'test_depth_pilot',
    'test_random_depth_pilot', 'test_random_depth_study', 'test_random_depth_migration',
    'test_retire_arwen', 'test_recovery_checkpoint', 'test_cluster',
    'test_analyze_radar_scaling_curve',
)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/dd2414-test-matplotlib')
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'cluster_rpl')]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in SUITES)
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests/unit')))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
