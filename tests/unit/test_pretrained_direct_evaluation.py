"""Reject substituted checkpoints or changed protocols before raw evaluation."""

import copy
import hashlib
import json
import unittest

from tools.evaluate_pretrained_checkpoint import TRAINER_COMMIT, validate_protocol


def payload():
    config = dict(source_code_commit=TRAINER_COMMIT, training_profile='random-metric-bev',
                  treatment='dino-only', mode='pretrain', epochs=50,
                  image_resolution=[448, 800], nsweeps=10, velocity_mode='full',
                  random_student_initialization=dict(kind='random'),
                  official_camera_initialization=None, train_samples=28130, val_samples=6019)
    return dict(run_config=config, checkpoint_kind='ddp_full_model',
                run_config_sha256=hashlib.sha256(json.dumps(
                    config, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


class DirectEvaluationTests(unittest.TestCase):
    def test_matching_pretraining_protocol(self):
        validate_protocol(payload(), 'dino-only')

    def test_finetuned_or_modified_protocol_cannot_masquerade_as_raw_pretraining(self):
        for key, replacement in [('mode', 'finetune'), ('epochs', 5),
                                 ('source_code_commit', 'different-source'),
                                 ('nsweeps', 5), ('val_samples', 64),
                                 ('random_student_initialization', dict(kind='pretrained'))]:
            value = payload()
            value['run_config'][key] = replacement
            value['run_config_sha256'] = hashlib.sha256(json.dumps(
                value['run_config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(value, 'dino-only')

    def test_integrity_and_treatment_must_match(self):
        value = payload()
        wrong_hash = copy.deepcopy(value)
        wrong_hash['run_config_sha256'] = 'corrupted'
        with self.assertRaises(ValueError):
            validate_protocol(wrong_hash, 'dino-only')
        with self.assertRaises(ValueError):
            validate_protocol(value, 'dino-metric-depth-bev')


if __name__ == '__main__':
    unittest.main()
