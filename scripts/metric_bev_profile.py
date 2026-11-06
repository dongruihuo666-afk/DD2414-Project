"""Admission and provenance for the opt-in random-initialization depth study."""
from pathlib import Path

from baseline_followup import digest_file, state_digest
from metric_depth_bev import (PROFILE, TREATMENTS, METRIC_WEIGHT_SHA256, build_metric_model,
    configure_metric_pretraining, load_metric_teacher)
from metric_bev_objective import MetricPretrainingForward


def validate_metric_args(args, environment):
    world = int(environment.get('WORLD_SIZE', '0'))
    if (world not in (2, 5, 10) or int(environment.get('LOCAL_WORLD_SIZE', '0')) != world
            or int(environment.get('RANK', '-1')) != int(environment.get('LOCAL_RANK', '-2'))):
        raise ValueError('Use 2, 5 or 10 local ranks; full 28130 samples must divide exactly, without padding/dropping')
    if (args.training_profile != PROFILE or args.treatment not in TREATMENTS
            or args.mode != 'pretrain' or args.depth_mode != 'metric-bev'
            or args.epochs != 50 or args.batch_size != 1 or args.precision != 'bf16'
            or args.nsweeps != 10 or args.velocity_mode != 'full'
            or args.train_samples not in (0, 28130) or args.val_samples != 6019
            or args.image_resolution != [448, 800] or args.teacher_resolution != [448, 798]
            or args.depth_teacher_resolution != [448, 798]):
        raise ValueError('Random metric-BEV study requires 50 epochs, full data, 10 sweeps, velocity, BF16 and 448x800 RGB')
    if (args.official_camera_checkpoint or args.pretrain_initial_checkpoint or args.pretrain_initial_sha256
            or args.checkpoint or args.initial_camera_probe or args.allow_source_relocation):
        raise ValueError('Random study forbids pretrained student initialization or continuation; --resume is the only restart')
    if ((args.seed, args.camera_lr, args.learning_rate) != (125, 2e-4, 2e-4)
            or (args.image_weight, args.bev_weight, args.motion_weight, args.depth_weight,
                args.relative_depth_weight) != (.2, .8, 0., 0., 0.)
            or args.metric_depth_sigma != 2. or args.metric_depth_weight != .05
            or not 0 <= args.num_workers <= 2
            or min(args.checkpoint_every, args.print_every) < 1 or args.stop_after_updates < 0):
        raise ValueError('Unmatched optimizer, loss, target-width or loader settings')
    if args.treatment == 'dino-metric-depth-bev':
        if not args.metric_teacher_source or not args.metric_teacher_weights or not args.metric_teacher_sha256:
            raise ValueError('Treatment requires explicit metric teacher source, weights and pinned SHA-256')
        if args.metric_teacher_sha256 != METRIC_WEIGHT_SHA256:
            raise ValueError('Use the pinned official outdoor metric teacher, not relative depth weights')


def prepare_metric(args, device):
    backbone = build_metric_model(device, args.seed, args.bevcar_source)
    initialization = dict(kind='random', seed=args.seed, pretrained_backbone=False,
                          student_state_sha256=state_digest(backbone))
    configure_metric_pretraining(backbone)
    supervised = args.treatment == 'dino-metric-depth-bev'
    teacher = (load_metric_teacher(args.metric_teacher_source, args.metric_teacher_weights,
               args.metric_teacher_sha256, device) if supervised else None)
    return backbone, MetricPretrainingForward(backbone, supervised), teacher, initialization


def metric_config(args, root, initialization):
    files = ['scripts/metric_depth_bev.py', 'scripts/metric_bev_objective.py',
             'scripts/metric_bev_profile.py', 'scripts/dinov2_bev_demo.py']
    result = dict(training_profile=PROFILE, random_student_initialization=initialization,
        camera_initialization='Random ResNet101; no ImageNet, Simple-BEV or prior student weights',
        pretraining_continuation=None, official_camera_initialization=None,
        losses='0.2 image DINO + 0.8 radar-supported fused BEV DINO; treatment adds 0.05 projected metric depth CE',
        depth_geometry=dict(bins=80, min_m=1., max_m=80., sigma_m=2.,
            loss_location='calibrated camera BEV volume before radar fusion',
            scope='Frozen outdoor metric pseudo-depth; not independent ground truth',
            inference='Keep student probability head; discard frozen teachers'),
        warmup_epochs=1., minimum_lr_fraction=.1, milestone_epochs=[5, 10, 20, 30, 40, 50],
        additional_implementation_sha256={name:digest_file(root/name) for name in files})
    if args.treatment == 'dino-metric-depth-bev':
        import hashlib
        metric = Path(args.metric_teacher_source)/'metric_depth'
        tree = hashlib.sha256()
        for path in sorted(metric.rglob('*.py')):
            tree.update(path.relative_to(metric).as_posix().encode()+b'\0')
            tree.update(path.read_bytes())
        result['metric_teacher'] = dict(weight_sha256=args.metric_teacher_sha256,
            dpt_sha256=digest_file(metric/'depth_anything_v2/dpt.py'),
            encoder_sha256=digest_file(metric/'depth_anything_v2/dinov2.py'),
            python_tree_sha256=tree.hexdigest(),
            maximum_depth_m=80, model='Depth-Anything-V2-Metric-VKITTI-Small')
    return result


def validate_metric_source(config, args, manifest, root):
    if (config.get('training_profile') != PROFILE or config.get('treatment') != args.treatment
            or config.get('random_student_initialization', {}).get('kind') != 'random'
            or config.get('official_camera_initialization') is not None):
        raise ValueError('Downstream requires matching random metric-BEV provenance and treatment')
    for name, expected in config['model_data_implementation_sha256'].items():
        if digest_file(root/name) != expected:
            raise ValueError('Metric-BEV source model/data implementation mismatch: ' + name)
    for name in ['scripts/metric_depth_bev.py', 'scripts/metric_bev_profile.py']:
        if digest_file(root/name) != config['additional_implementation_sha256'][name]:
            raise ValueError('Metric-BEV model implementation mismatch: ' + name)
