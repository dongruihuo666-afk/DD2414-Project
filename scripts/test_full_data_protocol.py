"""Leakage checks must reject unsafe partitions independently of model code."""
import unittest
from audit_full_data_protocol import experiment_tasks, historical_holdout_candidate, partition_errors


class FullDataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parts=dict(train=['train-a','train-b'],validation=['val-a'],test=['test-a'])

    def test_disjoint_labeled_unseen_split_can_pass(self):
        self.assertEqual(partition_errors(self.parts,['train-a'],test_has_labels=True),[])

    def test_pretrained_exposure_to_validation_or_test_is_rejected(self):
        for scene in ['val-a','test-a']:
            self.assertTrue(any('checkpoint' in e for e in
                partition_errors(self.parts,[scene],test_has_labels=True)))

    def test_previous_evaluation_precludes_blind_test_claim(self):
        self.assertTrue(any('blind test' in e for e in
            partition_errors(self.parts,historical_test_exposure=['test-a'],test_has_labels=True)))

    def test_unlabeled_test_cannot_be_used_for_required_metrics(self):
        self.assertTrue(any('annotations' in e for e in partition_errors(self.parts)))

    def test_overlap_and_missing_partition_fail(self):
        self.parts['test']=['train-a']
        self.assertTrue(any('overlap' in e for e in partition_errors(self.parts,test_has_labels=True)))
        del self.parts['test']
        self.assertTrue(any('no approved' in e for e in partition_errors(self.parts,test_has_labels=True)))

    def test_specification_has_eleven_pending_full_data_tasks(self):
        tasks=experiment_tasks()
        self.assertEqual(len(tasks),11)
        self.assertEqual(len({t['name'] for t in tasks}),11)
        self.assertTrue(all(t['status']=='pending_test_protocol' for t in tasks))
        self.assertTrue(all(t['configuration']['training_data']=='entire approved training partition'
                            for t in tasks))

    def test_candidate_keeps_whole_logs_and_is_not_activated(self):
        scenes=[dict(token=str(i),name=f'scene-{i}',log_token='a' if i<4 else 'b')
                for i in range(6)]
        samples=[dict(scene_token=str(i)) for i in range(6)]
        candidate=historical_holdout_candidate(scenes,samples,[s['name'] for s in scenes])
        self.assertFalse(candidate['active'])
        self.assertEqual(candidate['partitions']['validation']['scenes'],2)
        self.assertEqual(candidate['partitions']['final_holdout']['scenes'],4)
        self.assertEqual(candidate['partitions']['validation']['logs'],1)
        self.assertEqual(candidate['partitions']['final_holdout']['logs'],1)


if __name__=='__main__':
    unittest.main()
