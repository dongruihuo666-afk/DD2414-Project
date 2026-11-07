#!/usr/bin/env python3
"""Fail-closed audit before selecting a labeled, previously unseen test protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TEST_URL = ('https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/'
            'public/v1.0/v1.0-test_meta.tgz')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def partition_errors(partitions, initialization_scenes=(), historical_test_exposure=(),
                     test_has_labels=False):
    """Metadata disjointness alone is insufficient for a new blind test claim."""
    errors = []
    for name in ['train', 'validation', 'test']:
        records = partitions.get(name, [])
        if not records:
            errors.append(f'{name}: no approved scenes')
        elif len(records) != len(set(records)):
            errors.append(f'{name}: duplicate scenes')
    for left, right in [('train','validation'), ('train','test'), ('validation','test')]:
        if set(partitions.get(left, [])) & set(partitions.get(right, [])):
            errors.append(f'{left}/{right}: scene overlap')
    for name in ['validation','test']:
        if set(partitions.get(name, [])) & set(initialization_scenes):
            errors.append(f'{name}: checkpoint was trained on proposed held-out scenes')
    if set(partitions.get('test', [])) & set(historical_test_exposure):
        errors.append('test: previous evaluation exposure; cannot claim a new blind test')
    if not test_has_labels:
        errors.append('test: required vehicle annotations are unavailable')
    return errors


def experiment_tasks(official_protocol=False):
    """Requested scope, not executable commands or a claim that training has started."""
    common = dict(training_data='entire approved training partition',
        validation_data='entire approved validation partition',
        test_access='locked until protocol and model-selection policy are approved',
        camera_initialization='official Simple-BEV RGB encoder; strict load',
        image_resolution=[448,800], radar_sweeps=10, depth_mode='none')
    if official_protocol:
        common.update(training_data='official train: 700 scenes / 28130 samples',
            validation_data='official val: 150 scenes / 6019 samples',
            test_access='no local test; report validation metrics, never blind-test scores')
    tasks = []
    def add(name, mode, epochs, parent=None, velocity=None, location=None):
        tasks.append(dict(name=name,mode=mode,epochs=epochs,parent=parent,
            velocity_input=velocity,probe_location=location,
            status='queued' if official_protocol else 'pending_test_protocol',configuration=dict(common)))
    add('official_rgb_encoder_initial_probe','frozen-probe',10,location='camera_bev')
    for velocity in ['present','hidden_throughout_training_and_evaluation']:
        parent=f'double_dino_velocity_{velocity}'
        add(parent,'pretrain',5,velocity=velocity)
        for location in ['camera_bev','fused_bev']:
            add(f'{parent}_{location}_probe','frozen-probe',10,parent,velocity,location)
        add(f'{parent}_object_velocity','object-motion',5,parent,velocity)
        add(f'{parent}_segmentation_finetune','finetune',5,parent,velocity)
    return tasks


def historical_holdout_candidate(scenes, samples, validation_names, seed=125):
    """Propose a log-isolated split using metadata only; never activate it."""
    grouped={}
    for scene in scenes:
        if scene['name'] in set(validation_names):
            grouped.setdefault(scene['log_token'],[]).append(scene['token'])
    logs=sorted(grouped,key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    target=sum(map(len,grouped.values()))//2
    reachable={0:[]}
    for log in logs:
        for count,chosen in list(reachable.items()):
            new=count+len(grouped[log])
            if new<=target and new not in reachable:
                reachable[new]=chosen+[log]
    chosen=set(reachable[max(reachable)])
    roles={name:{s for log in logs if (log in chosen)==(name=='validation')
                    for s in grouped[log]} for name in ['validation','final_holdout']}
    return dict(active=False,
        limitation='Both partitions were historically evaluated; this is not a new blind test.',
        selection='Seeded metadata-only whole-log assignment; no outcomes used.',
        partitions={name:dict(scenes=len(tokens),
            samples=sum(s['scene_token'] in tokens for s in samples),
            logs=sum(bool(set(grouped[log]) & tokens) for log in logs))
            for name,tokens in roles.items()})


def audit(data_root, archive, checkpoint, official_protocol=False):
    from nuscenes.utils.splits import create_splits_scenes
    with (data_root/'v1.0-trainval/scene.json').open() as handle:
        scenes = json.load(handle)
    with (data_root/'v1.0-trainval/sample.json').open() as handle:
        samples = json.load(handle)
    official = create_splits_scenes()
    by_token = {s['token']: s for s in scenes}
    rows = {}
    scene_sets, log_sets, sample_sets = {}, {}, {}
    for name in ['train','val']:
        names = set(official[name])
        tokens = {s['token'] for s in scenes if s['name'] in names}
        scene_sets[name] = tokens
        log_sets[name] = {by_token[t]['log_token'] for t in tokens}
        sample_sets[name] = {s['token'] for s in samples if s['scene_token'] in tokens}
        rows[name] = dict(scenes=len(tokens),
            samples=len(sample_sets[name]),logs=len(log_sets[name]))
    manifest=json.loads((ROOT/'configs/radar_scaling_manifest_seed125.json').read_text())
    if set(manifest['scales']) != {'full'}:
        raise ValueError('Active manifest still contains subset definitions')
    if (set(manifest['scales']['full']['tokens']) != sample_sets['train'] or
            set(manifest['val']['tokens']) != sample_sets['val']):
        raise ValueError('Full manifest membership differs from official metadata')
    unsigned={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    digest=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=manifest['manifest_sha256']:
        raise ValueError('Active manifest hash is invalid')
    test = dict(download_url=TEST_URL, metadata_archive=str(archive),
                sensor_archives_downloaded=False, annotations_public=False)
    if archive.is_file():
        with tarfile.open(archive,'r:gz') as source:
            for table in ['scene','sample','sample_annotation','instance']:
                with source.extractfile(f'v1.0-test/{table}.json') as handle:
                    test[table+'_records'] = len(json.load(handle))
        test['metadata_archive_sha256'] = digest_file(archive)
        test['required_gt_available'] = (test['sample_annotation_records'] > 0
                                          and test['instance_records'] > 0)
    else:
        test['required_gt_available'] = False
        test['metadata_download_verified'] = False
    exposure_file = ROOT/'artifacts/meeting_20261016/evidence.json'
    previous = json.loads(exposure_file.read_text()) if exposure_file.is_file() else {}
    overlaps = dict(scenes=len(scene_sets['train'] & scene_sets['val']),
        logs=len(log_sets['train'] & log_sets['val']),
        samples=len(sample_sets['train'] & sample_sets['val']))
    if any(overlaps.values()):
        raise ValueError('Official training and validation partitions overlap')
    if official_protocol and (rows['train']['samples'] != 28130 or
            rows['val']['samples'] != 6019 or not checkpoint.is_file()):
        raise ValueError('Official protocol requires complete data membership and RGB checkpoint')
    return dict(status='ready_official_train_val' if official_protocol else 'blocked_test_protocol_decision',
        approved_protocol='official_train_val' if official_protocol else None,
        official_trainval=rows,
        active_full_manifest_sha256=digest,
        manifest_membership_matches_official=True,
        trainval_overlap=overlaps,
        official_test=test,
        camera_checkpoint=dict(path=str(checkpoint),exists=checkpoint.is_file(),
            sha256=digest_file(checkpoint) if checkpoint.is_file() else None,
            declared_pretraining_split='all 700 official train scenes',
            provenance_limit='Coverage is based on the official reproduction report; weights do not encode a training-scene ledger.'),
        previous_validation_evaluation=dict(source=str(exposure_file),
            reported_frames=previous.get('validation_samples'),
            cannot_relabel_as_untouched_test=bool(previous)),
        proposed_historical_holdout=historical_holdout_candidate(scenes,samples,official['val']),
        blockers=[] if official_protocol else [
            'Official test has no public instance/box annotations for local Vehicle BEV IoU or GT-box velocity EPE.',
            'Holding out official train scenes does not remove their prior exposure in the requested RGB encoder.',
            'Official val has already been evaluated in this project; a new split cannot restore a never-observed test claim.',
            'A labeled test protocol and treatment of historical/pretrained exposure must be chosen before automatic training.'],
        sources=[
            'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md',
            'https://registry.opendata.aws/motional-nuscenes/'],
        experiment_tasks=experiment_tasks(official_protocol),
        training_launched=False,
        note='Audit only. Runtime state is separately recorded in the official-protocol queue pipeline.json.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/home/students2026/datasets/nuscenes'))
    p.add_argument('--test-metadata-archive',type=Path,
        default=ROOT/'.cache/nuscenes/v1.0-test_meta.tgz')
    p.add_argument('--camera-checkpoint',type=Path,
        default=ROOT/'checkpoints/8x5_5e-4_rgb12_22:43:46/model-000025000.pth')
    p.add_argument('--output',type=Path)
    p.add_argument('--require-ready',action='store_true',help='Exit nonzero while test approval is missing.')
    p.add_argument('--protocol', choices=['pending','official_train_val'], default='pending')
    args=p.parse_args()
    result=audit(args.data_root,args.test_metadata_archive,args.camera_checkpoint,
                 args.protocol == 'official_train_val')
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    if args.require_ready and result['status'] != 'ready_official_train_val':
        raise SystemExit(2)


if __name__=='__main__':
    main()
