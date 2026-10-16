#!/usr/bin/env python3
"""Serial persistent queue: elapsed/stale time can WARN, never kill a worker."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fullsize_training import atomic_json_write

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {}


def should_stop_worker(stop_requested, elapsed_seconds, stale_seconds):
    """Only an explicit service/user stop, not a slow batch or long validation."""
    return bool(stop_requested)


def experiment_stages(args):
    python = sys.executable
    common = [python,'scripts/train_multisweep_teacher.py',
        '--data-root',str(args.data_root),'--bevcar-source',str(args.bevcar_source),
        '--manifest',str(ROOT/'configs/radar_scaling_manifest_seed125.json'),
        '--batch-size',str(args.batch_size),'--num-workers',str(args.num_workers),
        '--precision','bf16']
    stages = []
    if not getattr(args,'skip_official',False):
        for radar,checkpoint in [(False,'8x5_5e-4_rgb12_22:43:46'),(True,'8x5_5e-4_rad25_18:55:34')]:
            directory = args.study_dir/('official_camera_radar_5sweep_112x192' if radar else 'official_camera_112x192')
            command = [python,'scripts/eval_pretrained_baseline.py','--data-root',str(args.data_root),
                '--manifest',str(ROOT/'configs/radar_scaling_manifest_seed125.json'),
                '--checkpoint',str(ROOT/'checkpoints'/checkpoint/'model-000025000.pth'),
                '--run-dir',str(directory),'--height','112','--width','192',
                '--batch-size',str(args.batch_size),'--num-workers',str(args.num_workers)]
            if radar:
                command.append('--use-radar')
            stages.append(dict(name=directory.name,directory=str(directory),command=command,
                               requires=None,expected_summary=str(directory/'summary.json')))
    configurations = [(5,'none'),(10,'none'),(5,'radar'),(5,'radar-dav2')]
    selected = getattr(args,'configurations',None)
    if selected:
        selected = set(selected)
        configurations = [(sweeps,depth) for sweeps,depth in configurations
            if f'{sweeps}-{depth}' in selected]
    for sweeps,depth in configurations:
        label = f'double_dino_{sweeps}sweep_depth_{depth}'
        pretrain = args.study_dir/f'{label}_pretrain_{args.pretrain_epochs}epochs'
        for mode in ['pretrain','frozen-probe','finetune','unfrozen-probe','object-motion']:
            # Same linear head frozen/unfrozen is needed to avoid conflating
            # unfreezing with the much larger official decoder's capacity.
            directory = pretrain if mode == 'pretrain' else args.study_dir/f'{label}_{mode}_{args.downstream_epochs}epochs'
            epochs = args.pretrain_epochs if mode == 'pretrain' else args.downstream_epochs
            command = [*common,'--run-dir',str(directory),'--mode',mode,
                '--nsweeps',str(sweeps),'--depth-mode',depth,'--epochs',str(epochs)]
            if mode != 'pretrain':
                command += ['--checkpoint',str(pretrain/'checkpoints/latest.pt')]
            if mode == 'finetune':
                command += ['--learning-rate','1e-4']
            stages.append(dict(name=directory.name,directory=str(directory),command=command,
                requires=pretrain.name if mode != 'pretrain' else None,
                expected_summary=str(directory/'summary.json')))
    return stages


def report(study,state):
    rows = []
    motion_lines = []
    lines = ['# Multi-sweep double-distillation study', '',
        'All scientific runs use 28,130 train frames and the same 6,019 validation frames.',
        'Every comparison is single-seed and short-budget, not a convergence claim.',
        'Official original-resolution references (448x800) are in the previous study; this queue also evaluates those checkpoints at 112x192.',
        'New/legacy encoders use 112x192. A low-resolution evaluation of high-resolution-trained official weights is still not a matched training protocol.',
        'The depth treatment changes BOTH supervision and lifting; isolate those components before attributing a gain to depth loss alone.', '',
        '| Configuration | Status | Vehicle IoU (%) | Precision (%) | Recall (%) | F1 (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for task in state['plan']:
        result = read_json(task['expected_summary'])
        entry = state['stages'].get(task['name'],{})
        validation = result.get('final_validation') or {}
        if not validation:
            completed = result.get('progress',{}).get('validations',{})
            validation = completed[max(completed,key=int)] if completed else {}
        name = task['name']
        status = result.get('status') or entry.get('status','queued')
        rows.append(dict(name=name,status=status,final_validation=validation,
                         source=task['expected_summary']))
        values = [validation.get(key) for key in ['vehicle_iou','precision','recall','f1']]
        lines.append('| '+' | '.join([name,status,*['pending' if v is None else f'{100*v:.3f}' for v in values]])+' |')
        if validation.get('metrics'):
            motion_lines += ['',f'Object velocity, {name}:', '',
                '| Radar input | Objects | Velocity EPE (m/s) | Moving EPE (m/s) |',
                '| --- | ---: | ---: | ---: |']
            for mode,groups in validation['metrics'].items():
                val,moving = groups['all']['velocity_epe_mps'],groups['moving']['velocity_epe_mps']
                motion_lines.append(f'| {mode} | {groups["all"]["objects"]} | {val if val is not None else "n/a"} | {moving if moving is not None else "n/a"} |')
            motion_lines += ['', 'GT box positions are supplied; this is not detection or nuScenes NDS.', '']
    lines += motion_lines + ['', '## Monitoring and reliability', '',
        'The queue has no elapsed-time, log-idle or 24-hour kill rule. Long validation only emits a warning.',
        'Failed workers retry from the last durable update at most twice, then independent experiments continue.',
        'Dependent stages never use incomplete or failed pretraining. No automatic data/batch/epoch reduction.',
        'Check pipeline.json, each status.json/console.log, TensorBoard and checkpoint timestamps.',
        'A service stop requests a checkpoint at the next completed update; external power/storage failures remain possible.', '',
        'Legacy single-sweep results are historical controls, NOT a sweep-only control for these new objectives.',
        'Image distillation, camera unfreezing, projection heads, fixed recent target support and training budget differ from legacy runs.',
        'Radar-cell motion reconstruction remains a training auxiliary, not evidence of independent vehicle-motion accuracy.',
        'Depth-anchor errors use radar measurements, not independent dense depth ground truth.',
        'The DAV2 term supervises relative inverse-depth shape; radar provides metric scale.',
        'Five/ten-sweep pretraining keeps the teacher radar anchors at age <= 0.1 seconds in both runs.',
        'All accumulated sweeps still enter VoxelNet; the DINO target support is not expanded in the ten-sweep treatment.',
        'One epoch of a linear readout may underfit. Weak probe IoU does not by itself prove that representation learning failed.']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json_write(rows,study/'results.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--bevcar-source',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--pretrain-epochs',type=int,default=1)
    p.add_argument('--downstream-epochs',type=int,default=1)
    p.add_argument('--legacy-study',type=Path)
    p.add_argument('--configuration',dest='configurations',action='append',
        choices=['5-none','10-none','5-radar','5-radar-dav2'],
        help='Run only the selected configuration; repeat to select multiple configurations.')
    p.add_argument('--skip-official',action='store_true',
        help='Do not rerun the already-established low-resolution official checkpoints.')
    p.add_argument('--plan-only',action='store_true')
    args = p.parse_args()
    if min(args.batch_size,args.pretrain_epochs,args.downstream_epochs) < 1:
        raise ValueError('positive batch size and epoch budgets required')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != 'dd2414-mini-baseline':
        raise RuntimeError('baseline branch required; cluster branch must remain untouched')
    args.study_dir = args.study_dir.resolve()
    if args.plan_only:
        print(json.dumps(experiment_stages(args),indent=2))
        return
    args.study_dir.mkdir(parents=True,exist_ok=True)
    lock = (args.study_dir/'queue.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:
        signal.signal(sig,request_stop)
    path = args.study_dir/'pipeline.json'
    plan = experiment_stages(args)
    if args.legacy_study:
        # Finish the already-running historical job from its compatible saved
        # checkpoint. Do not launch new one-sweep jobs under the new protocol.
        directory = args.legacy_study/'finetune_1sweep_velocity_full_3epochs'
        cfg = read_json(directory/'config.json')
        if cfg and (directory/'checkpoints/latest.pt').is_file():
            command = [sys.executable,'scripts/train_baseline_downstream.py']
            options = ['data_root','bevcar_source','manifest','checkpoint','run_dir','mode',
                'epochs','batch_size','accumulation','num_workers','seed','precision',
                'camera_lr','learning_rate','probe_lr','weight_decay','train_samples','val_samples']
            for key in options:
                if key in cfg:
                    command += ['--'+key.replace('_','-'),str(cfg[key])]
            plan.insert(0,dict(name='finish_existing_historical_single_sweep_finetuning',
                directory=str(directory),command=command,requires=None,
                expected_summary=str(directory/'summary.json')))
        # Recover the prematurely stopped multi-sweep legacy probe, no new
        # single-sweep pretraining. The old script has no mid-epoch resume.
        directory = args.legacy_study/'frozen_1x1_5sweep_1epoch'
        command = [sys.executable,'scripts/train_fullsize_linear_probe.py',
            '--data-root',str(args.data_root),'--bevcar-source',str(args.bevcar_source),
            '--manifest',str(ROOT/'configs/radar_scaling_manifest_seed125.json'),
            '--checkpoint',str(ROOT/'fullsize_baseline/runs/full28130_sweeps5_seed125/checkpoints/epoch008.pt'),
            '--output-dir',str(directory),'--epochs','1','--batch-size','5',
            '--num-workers',str(args.num_workers)]
        plan.insert(1,dict(name='recover_legacy_five_sweep_frozen_probe',directory=str(directory),
                          command=command,requires=None,expected_summary=str(directory/'summary.json')))
    state = read_json(path) or dict(started_unix_time=time.time(),stages={},plan=plan)
    if state['plan'] != plan:
        raise ValueError('queue configuration changed; use a new study directory')
    state['status'] = 'running'
    atomic_json_write(state,path)
    report(args.study_dir,state)

    def publish_evidence():
        # A transient GitHub/SSH error cannot discard results or stop training.
        try:
            result = subprocess.run([sys.executable,'scripts/publish_multisweep_followup.py',
                '--study-dir',str(args.study_dir)],cwd=ROOT,capture_output=True,text=True)
            state['publication'] = dict(returncode=result.returncode,
                message=(result.stdout+result.stderr)[-3000:])
        except Exception as error:
            state['publication'] = dict(returncode=-1,message=f'{type(error).__name__}: {error}')
        atomic_json_write(state,path)
    for task in plan:
        name = task['name']
        summary = read_json(task['expected_summary'])
        entry = state['stages'].setdefault(name,dict(status='queued',attempts=0))
        if summary.get('status') == 'complete':
            entry.update(status='complete',reused=True)
            continue
        if STOP:
            break
        if task['requires'] and state['stages'].get(task['requires'],{}).get('status') != 'complete':
            entry.update(status='blocked_by_failed_pretraining')
            continue
        directory = Path(task['directory'])
        directory.mkdir(parents=True,exist_ok=True)
        for attempt in range(entry.get('attempts',0)+1,4):
            command = list(task['command'])
            if (directory/'checkpoints/latest.pt').is_file() and any(
                script in command for script in ['scripts/train_multisweep_teacher.py','scripts/train_baseline_downstream.py']):
                command += ['--resume']
            entry.update(status='running',attempts=attempt,command=command,started_unix_time=time.time())
            state.update(current_stage=name,updated_unix_time=time.time())
            atomic_json_write(state,path)
            print('START '+name+' '+json.dumps(command),flush=True)
            with (directory/'console.log').open('a') as handle:
                handle.write('\nCOMMAND '+json.dumps(command)+'\n')
                handle.flush()
                process = subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                                            start_new_session=True)
                entry['pid'] = process.pid
                signaled = False
                warned = 0.
                while process.poll() is None:
                    now = time.time()
                    child = read_json(directory/'status.json')
                    heartbeat = max(entry['started_unix_time'],child.get('updated_unix_time',0),
                        (directory/'console.log').stat().st_mtime,
                        (directory/'status.json').stat().st_mtime if (directory/'status.json').exists() else 0)
                    state.update(child_status=child,updated_unix_time=now,
                                 idle_warning_seconds=max(now-heartbeat,0))
                    atomic_json_write(state,path)
                    if now-heartbeat > 1800 and now-warned > 1800:
                        print(f'WARNING: {name} has no recent progress file; worker is NOT terminated',flush=True)
                        warned = now
                    if should_stop_worker(STOP,now-entry['started_unix_time'],now-heartbeat) and not signaled:
                        # No timeout escalation. The trainer may be in a full-val
                        # pass and will checkpoint at its next safe boundary.
                        process.send_signal(signal.SIGUSR1 if 'scripts/train_fullsize_linear_probe.py' not in command else signal.SIGTERM)
                        signaled = True
                    time.sleep(10)
            result = read_json(task['expected_summary'])
            complete = process.returncode == 0 and result.get('status') == 'complete'
            entry.update(status='complete' if complete else ('partial' if STOP else 'failed'),
                returncode=process.returncode,finished_unix_time=time.time())
            atomic_json_write(state,path)
            report(args.study_dir,state)
            publish_evidence()
            if complete or STOP:
                break
            print(f'Worker failed: {name}; preserving checkpoint, attempt {attempt}/3',flush=True)
        if STOP:
            break
    state.update(status='partial' if STOP else ('complete' if all(
        state['stages'].get(t['name'],{}).get('status') == 'complete' for t in plan)
        else 'finished_with_failures'),updated_unix_time=time.time())
    atomic_json_write(state,path)
    report(args.study_dir,state)
    publish_evidence()
    print('QUEUE_'+state['status'].upper(),flush=True)


if __name__ == '__main__':
    main()
