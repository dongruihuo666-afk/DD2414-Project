"""CPU/Gloo tests; no CUDA allocation, dataset, SSH, Slurm, or external writes."""
import copy
import ast
import datetime
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from distributed_training import (AcknowledgedDistributedSampler, SeededEpochDataset,
    any_rank, load_ddp_checkpoint, mean_scalars, rank_zero_call, save_ddp_checkpoint,
    validation_shard, require_completed_ddp_pretraining)
from train_depth_ddp import validate_args
from train_multisweep_teacher import parse_args


class RandomDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, torch.tensor([random.random(),np.random.rand(),torch.rand(()).item()])


def _worker(rank, directory, phase):
    torch.set_num_threads(1)
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    dist.init_process_group('gloo',rank=rank,world_size=2,
        init_method='file://'+str(Path(directory)/f'rendezvous-{phase}'),
        timeout=datetime.timedelta(seconds=60))
    try:
        device = torch.device('cpu')
        if phase == 'reference':
            # DDP's average local gradient equals the concatenated-batch gradient.
            torch.manual_seed(7)
            linear = nn.Linear(2,1)
            reference = copy.deepcopy(linear)
            wrapped = DistributedDataParallel(linear)
            local_x = torch.tensor([[1.+rank,2.-rank]])
            wrapped(local_x).square().mean().backward()
            reference(torch.tensor([[1.,2.],[2.,1.]])).square().mean().backward()
            for p,q in zip(linear.parameters(),reference.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-6)
            assert any_rank(rank == 1,device)
            assert mean_scalars(dict(loss=rank+1),device)['loss'] == 1.5
            def fail():
                raise ValueError('expected-test-error')
            try:
                rank_zero_call(fail)
            except RuntimeError as error:
                assert 'expected-test-error' in str(error)
            else:
                raise AssertionError('Rank-zero error was not propagated')
            # The real DDP wrapper must tolerate disabled depth parameters and
            # multiple heads returned in a dictionary over successive updates.
            from depth_pretraining_objective import PretrainingForward
            from test_depth_objective import TinyBackbone
            for mode in ['none','radar','radar-dav2']:
                backbone = TinyBackbone(mode)
                multihead = DistributedDataParallel(PretrainingForward(backbone),
                    find_unused_parameters=True,broadcast_buffers=True)
                opt = torch.optim.SGD(backbone.parameters(),lr=.01)
                for _ in range(2):
                    opt.zero_grad()
                    outputs = multihead(None)
                    loss = sum(value.square().mean() for value in outputs.values() if value is not None)
                    loss.backward()
                    opt.step()
                parameters = torch.cat([p.detach().flatten() for p in backbone.parameters()])
                reference_parameters = parameters.clone()
                dist.broadcast(reference_parameters,src=0)
                torch.testing.assert_close(parameters,reference_parameters,rtol=0,atol=0)
            local_indices = validation_shard(list(range(6019)),rank,2)
            validation_totals = torch.tensor([len(local_indices),sum(local_indices)],dtype=torch.float64)
            dist.all_reduce(validation_totals)
            assert validation_totals.tolist() == [6019,sum(range(6019))]
        random.seed(100+rank)
        np.random.seed(100+rank)
        torch.manual_seed(100+rank)
        model = nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Dropout(.2),nn.Linear(4,1))
        wrapped = DistributedDataParallel(model,broadcast_buffers=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        scaler = torch.amp.GradScaler('cuda',enabled=False)
        sampler = AcknowledgedDistributedSampler(range(12),125,rank,2)
        config = dict(world_size=2,global_batch_size=2,test='cpu-exact-resume')
        progress = dict(epoch=0,update=0,samples_seen=0)
        checkpoint = Path(directory)/'checkpoint.pt'
        if phase == 'resume':
            progress = load_ddp_checkpoint(checkpoint,model=model,optimizer=optimizer,
                scaler=scaler,sampler=sampler,config=config,device=device)
            assert progress['update'] == 2
        stop = 2 if phase == 'interrupted' else 3
        order = iter(sampler)
        while progress['update'] < stop:
            position = next(order)
            optimizer.zero_grad()
            x = torch.arange(8,dtype=torch.float32).reshape(2,4)/10+position/12
            # Different stochastic state per rank must survive a process restart.
            x = x+torch.rand(2,4)*.1+float(np.random.rand())*.01+random.random()*.01
            wrapped(x).square().mean().backward()
            optimizer.step()
            sampler.acknowledge(1)
            progress.update(epoch=sampler.epoch,update=progress['update']+1,
                            samples_seen=progress['samples_seen']+2)
        if phase == 'interrupted':
            save_ddp_checkpoint(checkpoint,model=model,optimizer=optimizer,scaler=scaler,
                sampler=sampler,progress=progress,config=config,device=device)
        if phase == 'resume':
            for bad in [dict(config,global_batch_size=4),dict(config,world_size=1)]:
                try:
                    load_ddp_checkpoint(checkpoint,model=model,optimizer=optimizer,scaler=scaler,
                        sampler=sampler,config=bad,device=device)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed resume configuration accepted')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(),progress=progress),Path(directory)/f'{phase}-rank{rank}.pt')
    finally:
        dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_resumed_learning_rate_telemetry_reads_restored_optimizer_groups(self):
        parameter = nn.Parameter(torch.ones(1))
        groups = [dict(params=[parameter],name='camera',lr=1e-5)]
        optimizer = torch.optim.AdamW(groups)
        state = optimizer.state_dict()
        state['param_groups'][0]['lr'] = 1.01e-6
        optimizer.load_state_dict(state)
        # load_state_dict replaces group dictionaries. Constructor-time groups
        # are stale and must never be used to report a resumed schedule.
        self.assertEqual(groups[0]['lr'],1e-5)
        tree = ast.parse(Path(__file__).with_name('train_depth_ddp.py').read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
            and ast.unparse(node.targets[0]) == "progress['latest']")
        expression = compile(ast.Expression(assignment.value),'<actual DDP telemetry>','eval')
        latest = eval(expression,dict(metrics={},groups=groups,optimizer=optimizer))
        self.assertEqual(latest['learning_rates']['camera'],1.01e-6)

    def test_full_training_shards_are_disjoint_complete_and_unpadded(self):
        samplers = [AcknowledgedDistributedSampler(range(28130),125,r,2) for r in range(2)]
        shards = [list(s) for s in samplers]
        self.assertEqual([len(s) for s in shards],[14065,14065])
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(set(shards[0]+shards[1]),set(range(28130)))
        self.assertEqual(samplers[0].position,0)  # iteration/prefetch is not acknowledgement
        for sampler in samplers:
            sampler.acknowledge(14065)
            self.assertEqual((sampler.epoch,sampler.position),(1,0))
        self.assertNotEqual(shards[0],list(samplers[0]))

    def test_sampler_refuses_padding_and_changed_rank_world_size(self):
        with self.assertRaises(ValueError):
            AcknowledgedDistributedSampler(range(7),125,0,2)
        first = AcknowledgedDistributedSampler(range(12),125,0,2)
        second = AcknowledgedDistributedSampler(range(12),125,1,2)
        first.acknowledge(2)
        state = first.state_dict()
        with self.assertRaises(ValueError):
            second.load_state_dict(state)
        restored = AcknowledgedDistributedSampler(range(12),125,0,2)
        restored.load_state_dict(state)
        self.assertEqual(list(first),list(restored))

    def test_odd_validation_split_is_complete_without_duplicate_padding(self):
        shards = [validation_shard(list(range(6019)),r,2) for r in range(2)]
        self.assertEqual([len(s) for s in shards],[3010,3009])
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(sorted(shards[0]+shards[1]),list(range(6019)))

    def test_sample_rng_survives_worker_prefetch_and_epoch_resume(self):
        dataset = SeededEpochDataset(RandomDataset(),125)
        torch.manual_seed(34)
        before = torch.get_rng_state().clone()
        first = dataset[4][1]
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(first,dataset[4][1],rtol=0,atol=0)
        dataset.epoch = 1
        self.assertFalse(torch.equal(first,dataset[4][1]))
        reference = list(DataLoader(dataset,batch_size=1,num_workers=0))
        prefetched = list(DataLoader(dataset,batch_size=1,num_workers=2))
        for a,b in zip(reference,prefetched):
            torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)

    def test_ddp_cli_is_explicit_and_rejects_changed_protocol(self):
        argv = ['train','--data-root','/data','--bevcar-source','/bevcar','--manifest','/manifest',
            '--run-dir','/run','--mode','pretrain','--distributed','--epochs','5','--nsweeps','10',
            '--batch-size','1','--precision','bf16','--image-resolution','448','800',
            '--teacher-resolution','448','798','--depth-teacher-resolution','448','798',
            '--pretrain-initial-checkpoint','/parent.pt','--pretrain-initial-sha256','hash',
            '--official-camera-checkpoint','/rgb.pt']
        with patch('sys.argv',argv):
            args = parse_args()
        env = dict(WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',RANK='0',LOCAL_RANK='0')
        validate_args(args,env)
        for key,value in [('mode','finetune'),('batch_size',2),('epochs',1),
                          ('image_resolution',[112,192]),('motion_weight',0),
                          ('pretrain_initial_checkpoint',None)]:
            bad = copy.copy(args)
            setattr(bad,key,value)
            with self.assertRaises(ValueError):
                validate_args(bad,env)
        with self.assertRaises(ValueError):
            validate_args(args,dict(env,WORLD_SIZE='1'))

    def test_two_process_gloo_gradient_sync_and_fresh_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            for phase in ['reference','interrupted','resume']:
                mp.spawn(_worker,args=(directory,phase),nprocs=2,join=True)
            for rank in range(2):
                a = torch.load(Path(directory)/f'reference-rank{rank}.pt',weights_only=False)
                b = torch.load(Path(directory)/f'resume-rank{rank}.pt',weights_only=False)
                self.assertEqual(a['sampler'],b['sampler'])
                self.assertEqual(a['progress'],b['progress'])
                for name,value in a['model'].items():
                    torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
                for index,state in a['optimizer']['state'].items():
                    for key,value in state.items():
                        torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)

    def test_downstream_refuses_last_training_update_without_full_validation(self):
        payload = dict(run_config=dict(mode='pretrain',epochs=5,train_samples=28130,
            val_samples=6019,distributed_training=dict(updates_per_epoch=14065)),
            progress=dict(epoch=5,update=70325,samples_seen=140650,
                validations={'5':dict(samples=6019)}),sampler=dict(epoch=5,position=0))
        require_completed_ddp_pretraining(payload)
        for section,key,value in [('progress','validations',{}),('progress','update',70324),
                                  ('sampler','position',1),('progress','samples_seen',140649)]:
            bad = copy.deepcopy(payload)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                require_completed_ddp_pretraining(bad)

    def test_preview_matrix_arguments_match_the_training_entrypoint(self):
        from cluster_rpl.depth_study import load_config, treatment_plan
        config = load_config(Path(__file__).resolve().parents[1]/'cluster_rpl/depth_study.example.json')
        env = dict(WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',RANK='0',LOCAL_RANK='0')
        for mode in ['none','radar','radar-dav2']:
            plan = treatment_plan(config,mode)
            with patch('sys.argv',plan['pretrain']['argv'][6:]):
                args = parse_args()
            validate_args(args,env)
            for readout in plan['downstream']:
                with patch('sys.argv',readout['argv'][1:]):
                    args = parse_args()
                self.assertFalse(args.distributed)
                self.assertIsNotNone(args.checkpoint)
                self.assertIsNone(args.pretrain_initial_checkpoint)
                self.assertEqual(args.depth_mode,mode)


if __name__ == '__main__':
    unittest.main()
