#!/usr/bin/env python3
"""Full-split, resumable baseline multi-sweep teacher and downstream training."""

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baseline_followup import (code_commit, digest_file, evaluate_predictions,
    labels_from_batch, load_datasets, state_digest, supervised_loss)
from dinov2_bev_demo import load_teacher
from fullsize_training import (AcknowledgedShuffleSampler, atomic_json_write,
    build_targets_from_features_batched, extract_teacher_features_batched,
    learning_rate_at_step, query_nvidia_smi, semantic_loss_batched,
    build_motion_targets_batched, motion_loss_batched)
from multisweep_teacher import (Downstream, build_model, configure,
    image_cosine_loss, load_depth_teacher, object_targets, pool_objects,
    prepare_bevcar_inputs, projected_radar_depth, relative_depth_loss,
    relative_depth_target, sparse_depth_loss, recent_radar_target_batch)
from radar_evaluation import RADAR_MODES, radar_for_mode, wrong_scene_dataset_indices
from radar_scaling_data import IndexedSubset, validate_scaling_manifest
from train_fullsize_linear_probe import reset_seeds
from train_nuscenes import SimpleLoss
from training_checkpoint import load_training_checkpoint, save_training_checkpoint

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f'SIGNAL {signum}: saving at the next update boundary', flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--bevcar-source', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--mode', choices=['pretrain','frozen-probe','unfrozen-probe',
                                    'finetune','object-motion'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--nsweeps', type=int, choices=[5,10], default=5)
    p.add_argument('--depth-mode', choices=['none','radar','radar-dav2'], default='none')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=125)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--camera-lr', type=float, default=1e-5)
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--image-weight', type=float, default=.2)
    p.add_argument('--bev-weight', type=float, default=.8)
    p.add_argument('--motion-weight', type=float, default=.5)
    p.add_argument('--depth-weight', type=float, default=.2)
    p.add_argument('--relative-depth-weight', type=float, default=.05)
    p.add_argument('--precision', choices=['bf16','fp16'], default='bf16')
    p.add_argument('--train-samples', type=int, default=0, help='smoke only; 0 = full split')
    p.add_argument('--val-samples', type=int, default=6019)
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--print-every', type=int, default=25)
    p.add_argument('--stop-after-updates', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


@torch.no_grad()
def object_predictions(model, inputs, dataset, tokens, device):
    features = model(inputs)
    predictions, records = [], []
    for bi, token in enumerate(tokens):
        objects = object_targets(dataset, token, inputs[3], device)
        values = pool_objects(features[bi:bi+1], objects)
        predictions.append(model.motion(values))
        records.append(objects)
    return predictions, records


@torch.no_grad()
def evaluate_motion(model, dataset, indices, manifest, args, device, dtype, output, status):
    """Same frozen backbone/readout, fixed GT locations, four radar inputs."""
    output.mkdir(parents=True, exist_ok=True)
    wrong_indices, mapping = wrong_scene_dataset_indices(manifest, args.seed)
    full_indices = manifest['val']['dataset_indices']
    wrong_by_index = dict(zip(full_indices, wrong_indices))
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(IndexedSubset(dataset, indices), **loader_args)
    wrong_loader = DataLoader(IndexedSubset(dataset, [wrong_by_index[i] for i in indices]),
                             **loader_args)
    sums = {mode: {group: dict(count=0, epe_sum=0., squared_sum=0., speed_error_sum=0.)
                  for group in ['all','moving','stationary']} for mode in RADAR_MODES}
    baselines = {name: dict(count=0, epe_sum=0.) for name in ['zero_prediction','train_mean']}
    frames = 0
    with (output/'per_object.jsonl').open('w') as handle:
        for (positions, batch), (_, wrong) in zip(loader, wrong_loader):
            matched = prepare_bevcar_inputs(batch, device)
            other = prepare_bevcar_inputs(wrong, device)[-1]
            tokens = [manifest['val']['tokens'][full_indices.index(indices[int(i)])]
                      for i in positions]
            for mode in RADAR_MODES:
                inputs = (*matched[:-1], radar_for_mode(matched[-1], mode, other))
                with torch.autocast('cuda', dtype=dtype):
                    predictions, records = object_predictions(model, inputs, dataset, tokens, device)
                for token, values, objects in zip(tokens, predictions, records):
                    for prediction, obj in zip(values.float(), objects):
                        target = obj['velocity']
                        error = float((prediction-target).norm())
                        speed_error = abs(float(prediction.norm())-float(target.norm()))
                        if not math.isfinite(error):
                            raise RuntimeError('non-finite object velocity prediction')
                        for group in ['all', 'moving' if obj['speed'] > 1 else 'stationary']:
                            row = sums[mode][group]
                            row['count'] += 1
                            row['epe_sum'] += error
                            row['squared_sum'] += error**2
                            row['speed_error_sum'] += speed_error
                        record = dict(sample_token=token, annotation_token=obj['annotation_token'],
                            radar_mode=mode, target_velocity_camera_xz_mps=target.tolist(),
                            prediction_velocity_camera_xz_mps=prediction.tolist(),
                            velocity_epe_mps=error, target_speed_mps=obj['speed'])
                        if mode == 'wrong_scene':
                            record['wrong_radar_sample_token'] = mapping[token]
                        handle.write(json.dumps(record)+'\n')
                        if mode == 'matched':
                            for name, value in [('zero_prediction', torch.zeros_like(target)),
                                ('train_mean', target.new_tensor(model.training_mean_velocity))]:
                                baselines[name]['count'] += 1
                                baselines[name]['epe_sum'] += float((value-target).norm())
            frames += len(positions)
            if frames % (args.batch_size*50) == 0 or frames == len(indices):
                handle.flush()
                status('validating_motion', validation_samples=frames)
                print(f'object_motion_validation={frames}/{len(indices)}', flush=True)
    metrics = {}
    for mode, groups in sums.items():
        metrics[mode] = {}
        for group, row in groups.items():
            n = row['count']
            metrics[mode][group] = dict(objects=n,
                velocity_epe_mps=row['epe_sum']/n if n else None,
                velocity_rmse_mps=math.sqrt(row['squared_sum']/n) if n else None,
                speed_mae_mps=row['speed_error_sum']/n if n else None)
    result = dict(samples=frames, metrics=metrics,
        baselines={k:dict(objects=v['count'], velocity_epe_mps=v['epe_sum']/max(v['count'],1))
                   for k,v in baselines.items()},
        protocol='Supervised linear readout of frozen BEV features at GT box locations; not detection/mAVE/NDS',
        targets='nuScenes box_velocity from temporal GT annotations; camera-XZ m/s; finite visible vehicles in ROI',
        wrong_scene_map='full validation cross-scene bijection; fixed camera, calibration and GT targets')
    atomic_json_write(result, output/'summary.json')
    return result


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.checkpoint_every, args.print_every) < 1:
        raise ValueError('epochs, batch size and intervals must be positive')
    if args.num_workers < 0 or not 0 <= args.train_samples <= 28130 or not 1 <= args.val_samples <= 6019:
        raise ValueError('invalid sample/worker settings')
    if min(args.image_weight,args.bev_weight,args.motion_weight,args.depth_weight,args.relative_depth_weight) < 0:
        raise ValueError('loss weights must be nonnegative')
    if args.mode == 'pretrain' and min(args.image_weight,args.bev_weight) <= 0:
        raise ValueError('double distillation requires both DINO terms')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU unit tests are not a training smoke')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; choose fp16 before locking the comparison')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:
        signal.signal(sig, request_stop)
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    reset_seeds(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    if args.seed != manifest['seed']:
        raise ValueError('seed differs from manifest')
    backbone = build_model(device, args.seed, args.bevcar_source, args.depth_mode)
    source = None
    if args.mode != 'pretrain':
        if args.checkpoint is None:
            raise ValueError('downstream requires a completed new double-distillation checkpoint')
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = source['run_config']
        if (cfg['experiment'] != 'baseline_multisweep_double_distillation'
                or cfg['mode'] != 'pretrain' or source['sampler']['epoch'] != cfg['epochs']):
            raise ValueError('source must be a completed double-distillation pretraining run')
        for key in ['nsweeps','depth_mode','seed']:
            if cfg[key] != getattr(args,key):
                raise ValueError(f'source {key} mismatch')
        if cfg['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('source manifest mismatch')
        if cfg['bevcar_sha256'] != digest_file(args.bevcar_source/'nets/voxelnet.py'):
            raise ValueError('source BEVCar implementation mismatch')
        backbone.load_state_dict(source['model'], strict=True)
        del source
    configure(backbone, args.mode)
    model = backbone if args.mode == 'pretrain' else Downstream(backbone, args.mode).to(device)
    frozen = args.mode in ['frozen-probe','object-motion']
    initial_digest = state_digest(backbone) if frozen else None
    datasets = load_datasets(args, manifest, args.nsweeps)
    count = args.train_samples or 28130
    indices = manifest['scales']['full']['dataset_indices'][:count]
    subset = IndexedSubset(datasets[0], indices)
    sampler = AcknowledgedShuffleSampler(subset, args.seed)
    teacher = load_teacher('dinov2_vits14', device) if args.mode == 'pretrain' else None
    depth_teacher = load_depth_teacher(device) if args.mode == 'pretrain' and args.depth_mode == 'radar-dav2' else None
    if teacher is not None and (teacher.training or any(p.requires_grad for p in teacher.parameters())):
        raise RuntimeError('DINO must remain frozen in eval mode')
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = []
    camera_ids = {id(p) for p in backbone.student.encoder.parameters()}
    head_ids = {id(p) for module in [getattr(model,'probe',None),getattr(model,'motion',None)]
                if module is not None for p in module.parameters()}
    for name, members, base_lr in [
        ('camera',[p for p in trainable if id(p) in camera_ids],
         args.learning_rate if args.mode == 'pretrain' else args.camera_lr),
        ('readout',[p for p in trainable if id(p) in head_ids],args.probe_lr),
        ('student',[p for p in trainable if id(p) not in camera_ids|head_ids],args.learning_rate)]:
        if members:
            groups.append(dict(params=members,lr=base_lr,base_lr=base_lr,name=name))
    optimizer = torch.optim.AdamW(groups, weight_decay=0 if frozen else 1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision=='fp16', init_scale=128.)
    criterion = SimpleLoss(2.13).to(device)
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ['resume','stop_after_updates','checkpoint_every','print_every']}
    config.update(experiment='baseline_multisweep_double_distillation',
        source_checkpoint_sha256=digest_file(args.checkpoint) if args.checkpoint else None,
        manifest_sha256=manifest['manifest_sha256'], train_samples=count,
        bevcar_sha256=digest_file(args.bevcar_source/'nets/voxelnet.py'),
        image_resolution=[112,192], camera_initialization='trainable random ResNet101; DINO image SSL',
        losses='image_weight * image DINO + bev_weight * radar-soft-masked BEV DINO + motion_weight * radar velocity reconstruction auxiliary; optional depth',
        segmentation_readout_loss='official SimpleLoss(pos_weight=2.13); not legacy class-balanced BCE',
        bev_target_radar_age_seconds=.1,
        radar_depth_target_age_seconds=.1,
        input_voxel_guard=7000,
        source_code_commit=code_commit(), frozen_initial_sha256=initial_digest,
        implementation_sha256={name:digest_file(ROOT/'scripts'/name) for name in
            ['multisweep_teacher.py','train_multisweep_teacher.py']},
        dino_weight_sha256=digest_file(ROOT/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'))
    if args.depth_mode == 'radar-dav2':
        config['dav2_weight_sha256'] = digest_file(ROOT/'.cache/depth_anything_v2_vits.pth')
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoints/latest.pt'
    progress = dict(epoch=0, update=0, samples_seen=0, training_seconds=0.,
        wall_seconds=0., validations={}, latest={}, object_velocity_sum=[0.,0.], object_count=0)
    if args.resume:
        config['source_code_commit'] = json.loads((directory/'config.json').read_text())['source_code_commit']
        progress = load_training_checkpoint(checkpoint, model=model, optimizer=optimizer,
            scaler=scaler, sampler=sampler, expected_run_config=config, map_location='cpu')
    elif checkpoint.exists() or (directory/'config.json').exists():
        raise FileExistsError('output exists; use --resume or a new directory')
    atomic_json_write(config, directory/'config.json')
    writer = SummaryWriter(str(directory/'tensorboard'), purge_step=progress['update'])
    start, prior = time.monotonic(), progress['wall_seconds']
    total_updates = math.ceil(count/args.batch_size)*args.epochs
    state = 'running'

    def status(state='running', **extra):
        speed = progress['samples_seen']/max(progress['training_seconds'],1e-9)
        value = dict(state=state, mode=args.mode, nsweeps=args.nsweeps,
            depth_mode=args.depth_mode, updated_unix_time=time.time(),
            epoch_progress=progress['samples_seen']/count, epochs=args.epochs,
            update=progress['update'], samples_seen=progress['samples_seen'],
            samples_per_second=speed, latest=progress['latest'],
            wall_seconds=prior+time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            eta_training_seconds=max(args.epochs*count-progress['samples_seen'],0)/max(speed,1e-9),
            throughput_scope='optimizer steps including input wait, excluding checkpoints and validation',
            gpu=query_nvidia_smi(), **extra)
        atomic_json_write(value, directory/'status.json')
        return value

    def save():
        progress['epoch'] = sampler.epoch
        progress['wall_seconds'] = prior+time.monotonic()-start
        save_training_checkpoint(checkpoint, model=model, optimizer=optimizer,
            scaler=scaler, sampler=sampler, progress=progress, run_config=config)

    def pretrain_loss(batch):
        target_batch = recent_radar_target_batch(batch)
        with torch.no_grad():
            features = extract_teacher_features_batched(teacher, batch[0][:,0], device)
            bev, mask = build_targets_from_features_batched(target_batch,features,device)
        # Target builders use inference_mode; normalize/clone outside it so that
        # autograd can save targets while training student projection heads.
        bev, mask = bev.clone(), mask.clone()
        inputs = prepare_bevcar_inputs(batch, device)
        shared, image, depth = backbone.features(inputs)
        parts = dict(image=image_cosine_loss(backbone.image_head(image),features),
            bev=semantic_loss_batched(backbone.bev_head(shared),bev,mask))
        loss = args.image_weight*parts['image'] + args.bev_weight*parts['bev']
        info = dict(bev_mask_cells=int((mask>0).sum()),
            radar_points_per_sample=(batch[16][:,0,:3].abs().sum(1)>0).sum(1).tolist(),
            occupied_voxels_per_sample=inputs[-1][-1].tolist())
        if args.motion_weight:
            motion, coverage, _ = build_motion_targets_batched(target_batch,device)
            parts['radar_motion_auxiliary'], cells = motion_loss_batched(
                backbone.motion_head(shared),motion,coverage)
            info['motion_auxiliary_covered_cells'] = cells
            loss = loss+args.motion_weight*parts['radar_motion_auxiliary']
        if depth is not None:
            anchors = projected_radar_depth(batch,*depth.shape[-2:],device)
            parts['radar_depth'], diagnostics = sparse_depth_loss(depth,anchors)
            info.update(diagnostics)
            loss = loss+args.depth_weight*parts['radar_depth']
            if depth_teacher is not None:
                with torch.no_grad():
                    target = relative_depth_target(depth_teacher,batch[0][:,0],device)
                parts['relative_depth'] = relative_depth_loss(depth,target)
                loss = loss+args.relative_depth_weight*parts['relative_depth']
        return loss, parts, info

    @torch.no_grad()
    def validate_pending():
        if sampler.position or sampler.epoch == 0 or str(sampler.epoch) in progress['validations']:
            return
        # Pretraining diagnostics and downstream GT metrics are kept distinct.
        model.eval()
        selected = manifest['val']['dataset_indices'][:args.val_samples]
        output = directory/f'validation/epoch{sampler.epoch:03d}'
        if args.mode == 'object-motion':
            model.training_mean_velocity = [v/max(progress['object_count'],1) for v in progress['object_velocity_sum']]
            result = evaluate_motion(model,datasets[1],selected,manifest,args,device,dtype,output,status)
        elif args.mode == 'pretrain':
            sums, samples, anchors, depth_abs, depth_sq, depth_rel = {},0,0,0.,0.,0.
            loader = DataLoader(IndexedSubset(datasets[1],selected),batch_size=args.batch_size,
                num_workers=args.num_workers,pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed))
            for _,batch in loader:
                with torch.autocast('cuda',dtype=dtype):
                    loss,parts,info = pretrain_loss(batch)
                n = len(batch[0])
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite pretraining validation')
                samples += n
                for key,value in dict(loss=loss,**parts).items():
                    sums[key] = sums.get(key,0.)+float(value)*n
                k = info.get('depth_anchors',0)
                if k:
                    anchors += k
                    depth_abs += k*info['depth_mae_m']
                    depth_sq += k*info['depth_rmse_m']**2
                    depth_rel += k*info['depth_abs_rel']
                if samples % (args.batch_size*100) == 0 or samples == len(selected):
                    status('validating_pretrain',validation_samples=samples)
                    print(f'pretrain_validation={samples}/{len(selected)}',flush=True)
            result = dict(samples=samples, losses={k:v/samples for k,v in sums.items()},
                depth_anchors=anchors, depth_mae_m=depth_abs/anchors if anchors else None,
                depth_rmse_m=math.sqrt(depth_sq/anchors) if anchors else None,
                depth_abs_rel=depth_rel/anchors if anchors else None,
                scope='Frozen-teacher / radar-anchor diagnostics, NOT segmentation accuracy or independent depth GT')
            atomic_json_write(result,output/'summary.json')
        else:
            def predict(batch):
                with torch.autocast('cuda',dtype=dtype):
                    outputs = model(prepare_bevcar_inputs(batch,device))
                return outputs[2] if args.mode == 'finetune' else outputs
            result = evaluate_predictions(predict,datasets[1],selected,args,manifest,output,
                lambda n: status('validating_segmentation',validation_samples=n))
        progress['validations'][str(sampler.epoch)] = result
        for key in ['vehicle_iou','precision','recall','f1']:
            if result.get(key) is not None:
                writer.add_scalar('validation/'+key,result[key],sampler.epoch)
        save()

    def train_mode():
        model.train()
        if args.mode == 'pretrain':
            backbone.student.decoder.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # A first-batch OOM/worker failure must have a compatible restart point,
        # not just a config.json that makes all subsequent attempts refuse.
        if not args.resume:
            save()
        validate_pending()
        status()
        with (directory/'metrics.jsonl').open('a') as log:
            while sampler.epoch < args.epochs and not STOP:
                train_mode()
                loader = DataLoader(subset,sampler=sampler,batch_size=args.batch_size,
                    num_workers=args.num_workers,pin_memory=True,
                    prefetch_factor=2 if args.num_workers else None,
                    generator=torch.Generator().manual_seed(args.seed+sampler.epoch))
                epoch_before = sampler.epoch
                iteration_clock = time.monotonic()
                for positions,batch in loader:
                    if STOP:
                        break
                    begin = iteration_clock
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate_at_step(group['base_lr'],progress['update'],total_updates,
                            round(math.ceil(count/args.batch_size)*.25),.1)
                    optimizer.zero_grad(set_to_none=True)
                    info = {}
                    with torch.autocast('cuda',dtype=dtype):
                        if args.mode == 'pretrain':
                            loss,parts,info = pretrain_loss(batch)
                        else:
                            inputs = prepare_bevcar_inputs(batch,device)
                            outputs = model(inputs)
                            if args.mode == 'object-motion':
                                values,targets = [],[]
                                for bi,position in enumerate(positions):
                                    token = manifest['scales']['full']['tokens'][int(position)]
                                    objects = object_targets(datasets[0],token,inputs[3],device)
                                    if objects:
                                        values.append(model.motion(pool_objects(outputs[bi:bi+1],objects)))
                                        targets.extend([obj['velocity'] for obj in objects])
                                if targets:
                                    truth = torch.stack(targets)
                                    prediction = torch.cat(values).float()
                                    loss = F.smooth_l1_loss(prediction,truth)
                                    info['training_velocity_sum'] = truth.detach().sum(0).tolist()
                                    info['training_objects'] = len(targets)
                                else:
                                    loss = sum(p.sum()*0 for p in model.motion.parameters())
                                parts = dict(object_velocity_smooth_l1=loss)
                            else:
                                labels = labels_from_batch(batch,device)
                                if args.mode == 'finetune':
                                    loss,parts,_ = supervised_loss(backbone.student,outputs,labels,criterion)
                                else:
                                    loss = criterion(outputs,labels['segmentation'],labels['valid'])
                                    parts = dict(segmentation_bce=loss)
                    if not torch.isfinite(loss):
                        raise RuntimeError('non-finite training loss; last completed checkpoint is preserved')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(trainable,5.,error_if_nonfinite=True)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < previous_scale:
                        raise RuntimeError('AMP overflow: no samples acknowledged; resume from durable checkpoint')
                    samples = len(positions)
                    sampler.acknowledge(samples)
                    progress['update'] += 1
                    progress['samples_seen'] += samples
                    progress['training_seconds'] += time.monotonic()-begin
                    progress['object_count'] += info.get('training_objects',0)
                    progress['object_velocity_sum'] = [a+b for a,b in zip(progress['object_velocity_sum'],
                        info.pop('training_velocity_sum',[0.,0.]))]
                    progress['latest'] = dict(loss=float(loss.detach()),gradient_norm=float(norm),
                        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                        **{k:float(v.detach()) for k,v in parts.items()},**info)
                    log.write(json.dumps(dict(update=progress['update'],epoch_progress=progress['samples_seen']/count,
                        samples_seen=progress['samples_seen'],**progress['latest']))+'\n')
                    for key,value in dict(loss=loss,**parts).items():
                        writer.add_scalar('train/'+key,float(value.detach()),progress['update'])
                    if progress['update'] % args.checkpoint_every == 0:
                        save()
                    if progress['update'] % args.print_every == 0:
                        print(json.dumps(status()),flush=True)
                        log.flush()
                        writer.flush()
                    if args.stop_after_updates and progress['update'] >= args.stop_after_updates:
                        state = 'partial'
                        break
                    iteration_clock = time.monotonic()
                del loader
                save()
                if sampler.epoch != epoch_before:
                    validate_pending()
                    save_training_checkpoint(directory/f'checkpoints/epoch{sampler.epoch:03d}.pt',
                        model=model,optimizer=optimizer,scaler=scaler,sampler=sampler,
                        progress=progress,run_config=config)
                if state == 'partial':
                    break
            state = 'complete' if sampler.epoch >= args.epochs else 'partial'
            save()
        if frozen and state_digest(backbone) != initial_digest:
            raise RuntimeError('frozen backbone parameters or BatchNorm buffers changed')
        telemetry = status(state)
        atomic_json_write(dict(status=state,config=config,progress=progress,telemetry=telemetry,
            final_validation=progress['validations'].get(str(args.epochs))),directory/'summary.json')
        print('MULTISWEEP_'+state.upper(),flush=True)
    except Exception as error:
        status('failed',message=f'{type(error).__name__}: {error}')
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
