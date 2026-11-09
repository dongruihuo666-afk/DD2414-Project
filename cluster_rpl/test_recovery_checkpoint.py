"""Runtime-only checkpoint copying and real 0/2-worker augmentation equivalence."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from recovery_checkpoint import canonical, file_digest, migrate, resume_digest
from distributed_training import SeededEpochDataset
import recover_random_depth as recovery


class RandomSamples(Dataset):
    def __len__(self): return 12
    def __getitem__(self,index):
        return torch.tensor([index,random.random(),np.random.rand(),torch.rand(()).item()])


class RecoveryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.source=self.root/'old.pt'
        config=dict(training_profile='random-metric-bev',treatment='dino-only',num_workers=2,
            run_dir=str(self.root/'old'),epochs=50,train_samples=28130,val_samples=6019,
            learning_rate=.0002,implementation_sha256={},model_data_implementation_sha256={},
            additional_implementation_sha256={})
        self.payload=dict(schema_version=2,checkpoint_kind='ddp_full_model',world_size=5,
            run_config=config,run_config_sha256=canonical(config),
            model=dict(weight=torch.arange(3.).to(torch.bfloat16),bn_count=torch.tensor(7)),
            optimizer=dict(state={0:dict(step=torch.tensor(2250.),exp_avg=torch.randn(3))}),
            progress=dict(update=2250,samples_seen=11250,epoch=0),sampler=dict(position=2250,epoch=0),
            distributed_state=[dict(rank=i,sampler=dict(position=2250,epoch=0),
                rng=dict(numpy=np.arange(8,dtype=np.uint32),torch=torch.arange(5,dtype=torch.uint8)),
                buffers=dict(mean=torch.zeros(3))) for i in range(5)])
        torch.save(self.payload,self.source)

    def test_real_round_trip_changes_only_worker_count_and_output_path(self):
        original=self.source.read_bytes()
        expected=resume_digest(self.payload)
        destination=self.root/'recovered'
        report=migrate(self.source,destination,self.root,2250,'dino-only')
        value=torch.load(destination/'checkpoints/latest.pt',weights_only=False)
        self.assertEqual(resume_digest(value),expected)
        self.assertEqual(report['unchanged_resume_state_sha256'],expected)
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(set(report['config_changes']),{'num_workers','run_dir'})
        self.assertEqual(value['run_config']['num_workers'],0)
        self.assertEqual(value['run_config_sha256'],canonical(value['run_config']))
        self.assertEqual(json.loads((destination/'config.json').read_text()),value['run_config'])

    def test_resume_hash_detects_optimizer_rng_model_and_cursor_changes(self):
        reference=resume_digest(self.payload)
        for field in ('model','optimizer','rng','cursor'):
            changed=copy.deepcopy(self.payload)
            if field=='model': changed['model']['weight'][0]=20
            elif field=='optimizer': changed['optimizer']['state'][0]['step']+=1
            elif field=='rng': changed['distributed_state'][3]['rng']['numpy'][0]+=1
            else: changed['progress']['update']+=1
            self.assertNotEqual(resume_digest(changed),reference)

    def test_invalid_config_budget_rank_state_or_nan_cannot_migrate(self):
        for mode in ('hash','budget','rank','nan','update'):
            value=copy.deepcopy(self.payload)
            if mode=='hash': value['run_config_sha256']='corrupted'
            elif mode=='budget':
                value['run_config']['epochs']=5
                value['run_config_sha256']=canonical(value['run_config'])
            elif mode=='rank': value['distributed_state'].pop()
            elif mode=='nan': value['optimizer']['state'][0]['exp_avg'][0]=float('nan')
            else: value['progress']['update']=2251
            torch.save(value,self.source)
            destination=self.root/('recovered-'+mode)
            with self.assertRaises(ValueError): migrate(self.source,destination,self.root,2250,'dino-only')
            self.assertFalse(destination.exists())

    def test_implementation_digest_mismatch_blocks_migration(self):
        (self.root/'model.py').write_text('original')
        self.payload['run_config']['model_data_implementation_sha256']={'model.py':file_digest(self.root/'model.py')}
        self.payload['run_config_sha256']=canonical(self.payload['run_config'])
        torch.save(self.payload,self.source)
        (self.root/'model.py').write_text('modified')
        with self.assertRaises(ValueError): migrate(self.source,self.root/'new',self.root,2250,'dino-only')

    def test_existing_destination_preserved(self):
        destination=self.root/'existing'
        destination.mkdir()
        with self.assertRaises(ValueError): migrate(self.source,destination,self.root,2250,'dino-only')

    def test_real_zero_and_two_worker_sample_outputs_are_identical(self):
        dataset=SeededEpochDataset(RandomSamples(),125)
        dataset.epoch=3
        results=[]
        for workers in (0,2):
            loader=DataLoader(dataset,batch_size=1,num_workers=workers,
                              generator=torch.Generator().manual_seed(25))
            results.append(torch.cat(list(loader)))
        torch.testing.assert_close(*results,rtol=0,atol=0)

    def test_recovery_command_keeps_old_training_source_and_full_protocol(self):
        cfg=json.loads((recovery.ROOT/'cluster_rpl/random_depth_study.example.json').read_text())
        cfg['workers_per_rank']=0
        cfg['recovery']=dict(training_root='/immutable/training')
        command=recovery.old_command(cfg,'dino-only')
        self.assertIn('/immutable/training/scripts/train_multisweep_teacher.py',command)
        for flag,value in [('--num-workers','0'),('--epochs','50'),('--batch-size','1'),('--nsweeps','10')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertEqual(command[command.index('--image-resolution')+1:command.index('--image-resolution')+3],['448','800'])

    def test_gate_requires_both_matching_real_resume_segments(self):
        cfg=dict(run_root=str(self.root),recovery=dict(expected_updates=dict(zip(recovery.TREATMENTS,(2250,3000)))))
        report=dict(status='passed',settings=cfg,controller_commit='controller',treatments={
            name:dict(status='passed',parent_update=start,pilot_initial_update=start,resumed_update=start+101,
                initial_segment=dict(exit_code=75,host_peak_gib=75.,host_limit_gib=128.),
                fresh_process_segment=dict(exit_code=75,host_peak_gib=76.,host_limit_gib=128.))
            for name,start in cfg['recovery']['expected_updates'].items()})
        path=self.root/'worker_free_recovery_passed.json'
        path.write_text(json.dumps(report))
        with patch.object(recovery,'commit',return_value='controller'):
            recovery.require_admission(cfg)
            report['treatments']['dino-only']['resumed_update']=2251
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): recovery.require_admission(cfg)

    def test_memory_retry_preserves_already_committed_pilot_progress(self):
        cfg=dict(run_root=str(self.root),recovery=dict(expected_updates=dict(zip(recovery.TREATMENTS,(2250,3000)))))
        report=dict(status='passed',settings=cfg,controller_commit='controller',treatments={
            name:dict(status='passed',parent_update=origin,pilot_initial_update=origin+101,resumed_update=origin+202,
                initial_segment=dict(exit_code=75,host_peak_gib=78.,host_limit_gib=128.),
                fresh_process_segment=dict(exit_code=75,host_peak_gib=79.,host_limit_gib=128.))
            for name,origin in cfg['recovery']['expected_updates'].items()})
        path=self.root/'worker_free_recovery_passed.json'
        path.write_text(json.dumps(report))
        with patch.object(recovery,'commit',return_value='controller'):
            recovery.require_admission(cfg)
            report['treatments']['dino-only']['initial_segment']['host_limit_gib']=80.
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): recovery.require_admission(cfg)


if __name__=='__main__': unittest.main()
