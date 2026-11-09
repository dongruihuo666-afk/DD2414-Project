"""Continuation must honor cancellation/OOM and validate final coverage."""
import unittest

from continue_recovered_depth import decision


def summary():
    return dict(status='partial',config=dict(training_profile='random-metric-bev',treatment='dino-only',
        epochs=50,num_workers=0,train_samples=28130,val_samples=6019),
        progress=dict(epoch=35,update=35*5626,samples_seen=35*28130),final_validation=None)


class ContinuationTests(unittest.TestCase):
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
