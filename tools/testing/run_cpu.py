import os
from pathlib import Path
import sys
import unittest

def main():
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/dd2414-independent-mpl')
    sys.path[:0]=[str(root/'src'),str(root),str(root/'scripts'),str(root/'cluster_rpl'),str(root/'cluster_experiment')]
    names=('test_fullsize_training','test_baseline_followup','test_multisweep_teacher',
        'test_metric_depth_bev','test_metric_memory','test_depth_objective','test_depth_continuation',
        'test_distributed_training','test_training_checkpoint','test_radar_evaluation','test_full_data_protocol',
        'test_radar_scaling_data','test_fullsize_linear_probe','test_trainval_configuration','test_official_full_data',
        'test_continue_recovered_depth','test_depth_assets','test_depth_study','test_depth_pilot',
        'test_random_depth_pilot','test_random_depth_study','test_random_depth_migration','test_retire_arwen',
        'test_recovery_checkpoint','test_cluster','test_analyze_radar_scaling_curve','test_hybrid_teacher_protocol','test_rpl_environment')
    selected=[n for n in names if any((root/folder/(n+'.py')).is_file() for folder in ('scripts','cluster_rpl','cluster_experiment'))]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in selected)
    if (root/'tests/unit').is_dir():suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests/unit')))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(not result.wasSuccessful())

if __name__=='__main__':main()
