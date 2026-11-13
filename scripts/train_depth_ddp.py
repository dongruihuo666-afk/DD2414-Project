"""Opt-in single-node depth pretraining: legacy continuation or random metric BEV.

No automatic Slurm submission. The old single-GPU/downstream trainer stays
available. Completed checkpoints expose unprefixed backbone tensors for it.
"""
from _bootstrap import enable_source_package
enable_source_package()
from dd2414.runtime.provenance import record_package_implementation

import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baseline_followup import code_commit, digest_file, load_datasets
from depth_pretraining_objective import PretrainingForward, pretraining_loss
from dinov2_bev_demo import load_teacher
from distributed_training import (AcknowledgedDistributedSampler, SeededEpochDataset, any_rank,
    load_ddp_checkpoint, mean_scalars, rank_zero_call, save_ddp_checkpoint, validation_shard)
from fullsize_training import atomic_json_write, learning_rate_at_step
from multisweep_teacher import build_model, configure, load_depth_teacher, load_official_camera_encoder
from radar_scaling_data import IndexedSubset, validate_scaling_manifest
from source_provenance import check_continuation_source
from train_fullsize_linear_probe import reset_seeds

ROOT = Path(__file__).resolve().parents[1]


def validate_args(args, environment):
    if getattr(args, 'training_profile', 'legacy-continuation') == 'random-metric-bev':
        from metric_bev_profile import validate_metric_args
        return validate_metric_args(args, environment)
    if (args.mode != 'pretrain' or args.batch_size != 1 or args.precision != 'bf16'
            or args.nsweeps != 10 or args.train_samples not in [0,28130]
            or args.val_samples != 6019 or args.epochs != 5
            or args.image_resolution != [448,800] or args.teacher_resolution != [448,798]
            or args.depth_teacher_resolution != [448,798]):
        raise ValueError('DDP study requires pretrain, 5 epochs, 10 sweeps, full data, BF16, '
                         'per-rank batch 1, RGB 448x800, both teachers 448x798')
    if (int(environment.get('WORLD_SIZE','0')) != 2
            or int(environment.get('LOCAL_WORLD_SIZE','0')) != 2
            or int(environment.get('RANK','-1')) != int(environment.get('LOCAL_RANK','-2'))):
        raise ValueError('Launch exactly two local processes with torchrun, on one node')
    if not args.pretrain_initial_checkpoint or not args.pretrain_initial_sha256 or not args.official_camera_checkpoint:
        raise ValueError('DDP continuation needs the pinned complete parent and official RGB provenance')
    if (args.checkpoint or args.initial_camera_probe or args.num_workers < 0
            or min(args.checkpoint_every,args.print_every) < 1 or args.stop_after_updates < 0
            or (args.seed,args.camera_lr,args.learning_rate,args.probe_lr) != (125,1e-5,2e-4,1e-3)
            or (args.image_weight,args.bev_weight,args.motion_weight,args.depth_weight,args.relative_depth_weight)
               != (.2,.8,.5,.2,.05)):
        raise ValueError('Unsupported or changed depth-study settings')


def run(args):
    validate_args(args, os.environ)
    metric_profile = getattr(args, 'training_profile', 'legacy-continuation') == 'random-metric-bev'
    world_size = int(os.environ['WORLD_SIZE'])
    updates_per_epoch = 28130 // world_size
    objective = pretraining_loss
    if metric_profile:
        from metric_bev_objective import metric_pretraining_loss
        objective = metric_pretraining_loss
    local_rank = int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError('Slurm-visible CUDA device count must match local ranks; CPU checks are not a full-model pilot')
    device = torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Native BF16 required on both ranks')
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    lock = None
    writer = None
    try:
        rank = dist.get_rank()
        directory = args.run_dir
        def own_output():
            nonlocal lock
            directory.mkdir(parents=True,exist_ok=True)
            lock = (directory/'.ddp.lock').open('w')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rank_zero_call(own_output)
        stopped = False
        def request_stop(signum, frame):
            nonlocal stopped
            stopped = True
        for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:
            signal.signal(sig,request_stop)
        stop_file = directory/'REQUEST_CHECKPOINT'
        reset_seeds(args.seed)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        manifest = json.loads(args.manifest.read_text())
        validate_scaling_manifest(manifest)
        if args.seed != manifest['seed']:
            raise ValueError('Manifest seed mismatch')
        if metric_profile:
            from metric_bev_profile import prepare_metric
            backbone, plain_forward, depth_teacher, initialization = prepare_metric(args, device)
            continuation = None
        else:
            backbone = build_model(device,args.seed,args.bevcar_source,args.depth_mode)
            initialization = load_official_camera_encoder(backbone.student.encoder,args.official_camera_checkpoint)
            if digest_file(args.pretrain_initial_checkpoint) != args.pretrain_initial_sha256:
                raise ValueError('Parent SHA-256 mismatch')
            source = torch.load(args.pretrain_initial_checkpoint,map_location='cpu',weights_only=False)
            continuation = check_continuation_source(source,dict(velocity_mode=args.velocity_mode),
                manifest['manifest_sha256'],args.official_camera_checkpoint,initialization,args.allow_source_relocation)
            if source['run_config']['bevcar_sha256'] != digest_file(args.bevcar_source/'nets/voxelnet.py'):
                raise ValueError('Parent BEVCar implementation mismatch')
            for name,value in source['run_config']['model_data_implementation_sha256'].items():
                if digest_file(ROOT/name) != value:
                    raise ValueError('Parent model/data implementation mismatch: '+name)
            backbone.load_state_dict(source['model'],strict=True)
            del source
            continuation.update(sha256=args.pretrain_initial_sha256,
                path=str(args.pretrain_initial_checkpoint.resolve()),additional_epochs=5,
                total_pretraining_epochs=10)
            configure(backbone,'pretrain')
            plain_forward = PretrainingForward(backbone)
            depth_teacher = load_depth_teacher(device) if args.depth_mode == 'radar-dav2' else None
        forward = DistributedDataParallel(plain_forward,device_ids=[local_rank],
            find_unused_parameters=True,broadcast_buffers=True)
        teacher = load_teacher('dinov2_vits14',device)
        for model in [teacher,depth_teacher]:
            if model is not None and (model.training or any(p.requires_grad for p in model.parameters())):
                raise RuntimeError('All teachers must remain frozen and in eval mode')
        datasets = load_datasets(args,manifest,10,resolution=args.image_resolution)
        for dataset in datasets:
            dataset.data_aug_conf['checkpoint_era_resize'] = True
        # Initialization is shared, but stochastic training streams should not
        # force identical dropout masks on the two different local examples.
        reset_seeds(args.seed+rank)
        subset = SeededEpochDataset(IndexedSubset(datasets[0],manifest['scales']['full']['dataset_indices']),args.seed)
        sampler = AcknowledgedDistributedSampler(subset,args.seed,rank,world_size)
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        camera_ids = {id(p) for p in backbone.student.encoder.parameters()}
        groups = []
        for name,members,rate in [('camera',[p for p in trainable if id(p) in camera_ids],args.camera_lr),
                                  ('student',[p for p in trainable if id(p) not in camera_ids],args.learning_rate)]:
            groups.append(dict(params=members,lr=rate,base_lr=rate,name=name))
        optimizer = torch.optim.AdamW(groups,weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda',enabled=False)
        config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
                  if k not in ['resume','stop_after_updates','checkpoint_every','print_every']}
        config.update(experiment='baseline_multisweep_double_distillation',
            source_code_commit=code_commit(),manifest_sha256=manifest['manifest_sha256'],
            train_samples=28130,val_samples=6019,official_camera_initialization=initialization,
            pretraining_continuation=continuation,
            camera_initialization='Complete five-epoch double-distillation parent; strict model-only initialization',
            bevcar_sha256=digest_file(args.bevcar_source/'nets/voxelnet.py'),
            dino_weight_sha256=digest_file(ROOT/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'),
            image_resize='Pillow RGB default (checkpoint-era)',
            data_protocol='official 700-scene train / 150-scene validation; no local blind test claim',
            losses='mean of rank-local per-sample objectives; image .2 + masked BEV .8 + radar motion .5; optional depth',
            bev_target_radar_age_seconds=.1,radar_depth_target_age_seconds=.1,input_voxel_guard=7000,
            distributed_training=dict(world_size=world_size,local_batch_size=1,global_batch_size=world_size,
                updates_per_epoch=updates_per_epoch,backend='nccl',padding=False,dropped_samples=False,
                sample_rng='seed+1000003*epoch+7919*position; isolated CPU RNG per sample',
                initial_training_rng='seed+rank; restored per rank on resume',
                batch_norm='per-rank statistics; buffers broadcast from rank zero before each train forward'),
            implementation_sha256={name:digest_file(ROOT/'scripts'/name) for name in
                ['train_depth_ddp.py','distributed_training.py','depth_pretraining_objective.py',
                 'train_multisweep_teacher.py','multisweep_teacher.py','baseline_followup.py',
                 'source_provenance.py','fullsize_training.py','radar_scaling_data.py',
                 'radar_evaluation.py','training_checkpoint.py']},
            model_data_implementation_sha256={name:digest_file(ROOT/name) for name in
                ['nets/segnet.py','nets/bevcar_voxel_adapter.py','nets/radar_encoder.py',
                 'nuscenesdataset.py','utils/vox.py','utils/geom.py']})
        if metric_profile:
            from metric_bev_profile import metric_config
            config.update(metric_config(args, ROOT, initialization))
        elif depth_teacher is not None:
            config['dav2_weight_sha256'] = digest_file(ROOT/'.cache/depth_anything_v2_vits.pth')
        record_package_implementation(config, ROOT)
        checkpoint = directory/'checkpoints/latest.pt'
        progress = dict(epoch=0,update=0,samples_seen=0,training_seconds=0.,wall_seconds=0.,
                        validations={},latest={})
        if args.resume:
            config['source_code_commit'] = json.loads((directory/'config.json').read_text())['source_code_commit']
            progress = load_ddp_checkpoint(checkpoint,model=backbone,optimizer=optimizer,scaler=scaler,
                sampler=sampler,config=config,device=device)
        elif any_rank(checkpoint.exists() or (directory/'config.json').exists(),device):
            raise FileExistsError('Output exists; choose --resume or a new directory')
        rank_zero_call(lambda: atomic_json_write(config,directory/'config.json'))
        if rank == 0:
            writer = SummaryWriter(str(directory/'tensorboard'),purge_step=progress['update'])
        start,prior = time.monotonic(),progress['wall_seconds']
        total_updates = updates_per_epoch*args.epochs

        def requested():
            return any_rank(stopped or (rank == 0 and stop_file.exists()),device)

        def status(state='running',**extra):
            if rank != 0:
                return None
            value = dict(state=state,mode='pretrain',nsweeps=10,depth_mode=args.depth_mode,
                training_profile=getattr(args,'training_profile','legacy-continuation'),
                treatment=getattr(args,'treatment',None),
                updated_unix_time=time.time(),epoch_progress=progress['samples_seen']/28130,
                epochs=args.epochs,update=progress['update'],samples_seen=progress['samples_seen'],
                samples_per_second=progress['samples_seen']/max(progress['training_seconds'],1e-9),
                latest=progress['latest'],wall_seconds=prior+time.monotonic()-start,
                total_updates=total_updates,
                world_size=world_size,global_batch_size=world_size,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,**extra)
            if metric_profile:
                value['memory_all_ranks_at_last_save'] = progress.get('memory_all_ranks_at_last_save')
            atomic_json_write(value,directory/'status.json')
            return value

        def save():
            progress['epoch'] = sampler.epoch
            elapsed = rank_zero_call(lambda: prior+time.monotonic()-start)
            progress['wall_seconds'] = elapsed
            if metric_profile:
                local_memory = dict(rank=rank,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                    device_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30)
                memory = [None]*world_size
                dist.all_gather_object(memory, local_memory)
                progress['memory_all_ranks_at_last_save'] = memory
            save_ddp_checkpoint(checkpoint,model=backbone,optimizer=optimizer,scaler=scaler,
                sampler=sampler,progress=progress,config=config,device=device)

        @torch.no_grad()
        def validate_pending():
            if sampler.position or not sampler.epoch or str(sampler.epoch) in progress['validations']:
                return True
            backbone.eval()
            # Every validation frame is scored once. Broadcast BN buffers so
            # both shards evaluate the SAME rank-zero checkpoint representation.
            for buffer in backbone.buffers():
                dist.broadcast(buffer,src=0)
            selected = manifest['val']['dataset_indices']
            local_indices = validation_shard(selected,rank,world_size)
            loader = DataLoader(IndexedSubset(datasets[1],local_indices),batch_size=1,
                num_workers=args.num_workers,pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed))
            iterator = iter(loader)
            sums,samples,anchors,absolute,squared,relative = {},0,0,0.,0.,0.
            keys = ['loss','image','bev'] if metric_profile else ['loss','image','bev','radar_motion_auxiliary']
            if not metric_profile and args.depth_mode != 'none':
                keys.append('radar_depth')
            if depth_teacher is not None:
                keys.append('metric_depth_bev' if metric_profile else 'relative_depth')
            for step in range(math.ceil(len(selected)/world_size)):
                if requested():
                    return False
                loss = None
                if step < len(local_indices):
                    _,batch = next(iterator)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss,parts,info = objective(batch,plain_forward,teacher,depth_teacher,args,device)
                    samples += 1
                    for key,value in dict(loss=loss,**parts).items():
                        sums[key] = sums.get(key,0.)+float(value)
                    count = info.get('depth_anchors',0)
                    if count:
                        anchors += count
                        absolute += count*info['depth_mae_m']
                        squared += count*info['depth_rmse_m']**2
                        relative += count*info['depth_abs_rel']
                if any_rank(loss is not None and not bool(torch.isfinite(loss)),device):
                    raise RuntimeError('Non-finite full validation loss')
                if step % 100 == 0:
                    status('validating_pretrain',validation_samples=min((step+1)*world_size,len(selected)))
            totals = torch.tensor([sums.get(k,0.) for k in keys]+[samples,anchors,absolute,squared,relative],
                                  device=device,dtype=torch.float64)
            dist.all_reduce(totals)
            values = totals.cpu().tolist()
            samples,anchors,absolute,squared,relative = values[-5:]
            if int(samples) != 6019:
                raise RuntimeError('Full validation coverage mismatch')
            result = dict(samples=int(samples),losses={k:v/samples for k,v in zip(keys,values)},
                depth_anchors=int(anchors),depth_mae_m=absolute/anchors if anchors else None,
                depth_rmse_m=math.sqrt(squared/anchors) if anchors else None,
                depth_abs_rel=relative/anchors if anchors else None,
                scope='Held-out teacher objective diagnostics; NOT segmentation IoU or independent dense GT')
            rank_zero_call(lambda: atomic_json_write(result,directory/f'validation/epoch{sampler.epoch:03d}/summary.json'))
            progress['validations'][str(sampler.epoch)] = result
            save()
            # Also retain milestones when resuming directly into pending
            # validation after a time-limit checkpoint at the epoch boundary.
            if not metric_profile or sampler.epoch in (5,10,20,30,40,50):
                rank_zero_call(lambda: _copy_epoch(checkpoint,directory,sampler.epoch))
            return True

        torch.cuda.reset_peak_memory_stats(device)
        if not args.resume:
            save()
        state = 'running'
        if not validate_pending():
            state = 'partial'
        while sampler.epoch < args.epochs and state == 'running':
            forward.train()
            backbone.student.decoder.eval()
            before = sampler.epoch
            subset.epoch = sampler.epoch
            loader = DataLoader(subset,sampler=sampler,batch_size=1,num_workers=args.num_workers,
                pin_memory=True,prefetch_factor=2 if args.num_workers else None,
                generator=torch.Generator().manual_seed(args.seed+sampler.epoch+rank*1000))
            iteration_clock = time.monotonic()
            for positions,batch in loader:
                if requested():
                    state = 'partial'
                    break
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate_at_step(group['base_lr'],progress['update'],total_updates,
                                                        round(updates_per_epoch*(1. if metric_profile else .25)),.1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,parts,info = objective(batch,forward,teacher,depth_teacher,args,device)
                if any_rank(not bool(torch.isfinite(loss)),device):
                    raise RuntimeError('At least one DDP rank has non-finite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable,5.,error_if_nonfinite=False)
                if any_rank(not bool(torch.isfinite(norm)),device):
                    raise RuntimeError('At least one DDP rank has non-finite gradient norm')
                optimizer.step()
                sampler.acknowledge(len(positions))
                metrics = mean_scalars(dict(loss=loss.detach(),gradient_norm=norm,**parts),device)
                elapsed = torch.tensor(time.monotonic()-iteration_clock,device=device)
                dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                progress['training_seconds'] += float(elapsed)
                progress['update'] += 1
                progress['samples_seen'] += world_size*len(positions)
                progress['latest'] = dict(metrics,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                if rank == 0:
                    with (directory/'metrics.jsonl').open('a') as log:
                        log.write(json.dumps(dict(update=progress['update'],
                            epoch_progress=progress['samples_seen']/28130,**progress['latest']))+'\n')
                    for key,value in metrics.items():
                        writer.add_scalar('train/'+key,value,progress['update'])
                if progress['update'] % args.checkpoint_every == 0:
                    save()
                if progress['update'] % args.print_every == 0 and rank == 0:
                    print(json.dumps(status()),flush=True)
                    writer.flush()
                if args.stop_after_updates and progress['update'] >= args.stop_after_updates:
                    state = 'partial'
                    break
                iteration_clock = time.monotonic()
            del loader
            save()
            if sampler.epoch != before:
                if not validate_pending():
                    state = 'partial'
        if sampler.epoch >= args.epochs and str(args.epochs) in progress['validations']:
            state = 'complete'
        else:
            state = 'partial'
        save()
        def finish():
            telemetry = status(state)
            atomic_json_write(dict(status=state,config=config,progress=progress,telemetry=telemetry,
                final_validation=progress['validations'].get(str(args.epochs))),directory/'summary.json')
            print('DISTRIBUTED_'+state.upper(),flush=True)
        rank_zero_call(finish)
    except Exception as error:
        if dist.get_rank() == 0:
            atomic_json_write(dict(state='failed',message=f'{type(error).__name__}: {error}',
                                  updated_unix_time=time.time()),args.run_dir/'status.json')
        raise
    finally:
        if writer is not None:
            writer.close()
        if lock is not None:
            lock.close()
        dist.destroy_process_group()


def _copy_epoch(checkpoint, directory, epoch):
    import shutil
    destination = directory/f'checkpoints/epoch{epoch:03d}.pt'
    temporary = destination.with_suffix('.pt.tmp')
    shutil.copyfile(checkpoint,temporary)
    temporary.replace(destination)
