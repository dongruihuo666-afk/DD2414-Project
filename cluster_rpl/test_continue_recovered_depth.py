"""Continuation must honor cancellation/OOM and validate final coverage."""
import unittest
import os
import pwd
from unittest.mock import patch

from continue_recovered_depth import decision, parent_accounting, slurm_query_environment


def summary():
    return dict(status='partial',config=dict(training_profile='random-metric-bev',treatment='dino-only',
        epochs=50,num_workers=0,train_samples=28130,val_samples=6019),
        progress=dict(epoch=35,update=35*5626,samples_seen=35*28130),final_validation=None)


class ContinuationTests(unittest.TestCase):
    def test_library_paths_are_scoped_preserve_existing_paths_and_leave_environment_unchanged(self):
        with patch.dict(os.environ, {'LD_LIBRARY_PATH':'/existing:/usr/local/lib'},clear=False):
            before=dict(os.environ)
            env=slurm_query_environment()
            self.assertEqual(env['LD_LIBRARY_PATH'],'/usr/local/lib:/usr/local/lib/slurm:/existing')
            self.assertEqual(dict(os.environ),before)

    def test_query_pins_command_libraries_job_owner_and_timeout(self):
        owner=pwd.getpwuid(os.getuid()).pw_name
        with patch('continue_recovered_depth.subprocess.check_output',return_value=f'42|{owner}|FAILED|75:0\n') as query:
            result=parent_accounting('42')
        self.assertEqual(result,dict(job_id='42',user=owner,state='FAILED',exit_code='75:0'))
        args,kwargs=query.call_args
        self.assertEqual(args[0][0],'/usr/local/bin/sacct')
        self.assertEqual(args[0][args[0].index('-j')+1],'42')
        self.assertEqual(kwargs['timeout'],10)
        self.assertTrue(kwargs['env']['LD_LIBRARY_PATH'].startswith('/usr/local/lib:/usr/local/lib/slurm'))

    def test_wrong_job_owner_or_multiple_rows_rejected(self):
        owner=pwd.getpwuid(os.getuid()).pw_name
        for output in (f'43|{owner}|FAILED|75:0', '42|another-owner|FAILED|75:0',
                       f'42|{owner}|FAILED|75:0\n42.batch|{owner}|FAILED|75:0'):
            with patch('continue_recovered_depth.subprocess.check_output',return_value=output):
                with self.assertRaises(ValueError): parent_accounting('42')

    def test_waits_for_terminal_accounting_then_returns_exact_parent(self):
        owner=pwd.getpwuid(os.getuid()).pw_name
        with patch('continue_recovered_depth.subprocess.check_output',side_effect=[
                '',f'42|{owner}|COMPLETING|0:0',f'42|{owner}|FAILED|75:0']), \
                patch('continue_recovered_depth.time.sleep') as sleep:
            self.assertEqual(parent_accounting('42')['exit_code'],'75:0')
            self.assertEqual(sleep.call_count,2)

    def test_missing_terminal_accounting_is_bounded(self):
        with patch('continue_recovered_depth.subprocess.check_output',return_value='') as query, \
                patch('continue_recovered_depth.time.sleep'):
            with self.assertRaises(RuntimeError): parent_accounting('42')
        self.assertEqual(query.call_count,30)

    def test_only_committed_exit_75_resumes(self):
        self.assertEqual(decision('FAILED','75:0',summary(),'dino-only'),'resume')

    def test_oom_cancellation_and_arbitrary_errors_never_retry(self):
        for state,code in [('OUT_OF_MEMORY','0:125'),('CANCELLED','75:0'),('FAILED','1:0'),('TIMEOUT','0:9')]:
            with self.assertRaises(RuntimeError): decision(state,code,summary(),'dino-only')

    def test_complete_requires_fifty_epochs_and_full_final_validation(self):
        value=summary()
        value.update(status='complete',progress=dict(epoch=50,update=50*5626,samples_seen=50*28130),
                     final_validation=dict(samples=6019))
        self.assertEqual(decision('COMPLETED','0:0',value,'dino-only'),'complete')
        value['final_validation']['samples']=64
        with self.assertRaises(ValueError): decision('COMPLETED','0:0',value,'dino-only')

    def test_final_validation_pending_can_resume_without_extra_epochs(self):
        value=summary()
        value['progress']=dict(epoch=50,update=50*5626,samples_seen=50*28130)
        self.assertEqual(decision('FAILED','75:0',value,'dino-only'),'resume')

    def test_wrong_treatment_workers_or_progress_rejected(self):
        for field in ('treatment','workers','progress'):
            value=summary()
            if field=='treatment': value['config']['treatment']='dino-metric-depth-bev'
            elif field=='workers': value['config']['num_workers']=2
            else: value['progress']['samples_seen']+=1
            with self.assertRaises(ValueError): decision('FAILED','75:0',value,'dino-only')

    def test_out_of_budget_or_inconsistent_epoch_rejected(self):
        for update,epoch in [(-1,0),(50*5626+1,50),(35*5626,36)]:
            value=summary()
            value['progress'].update(update=update,epoch=epoch,samples_seen=5*update)
            with self.assertRaises(ValueError): decision('FAILED','75:0',value,'dino-only')


if __name__=='__main__': unittest.main()
