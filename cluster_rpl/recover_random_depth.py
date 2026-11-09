#!/usr/bin/env python3
"""Worker-free DDP recovery; old immutable training code and all learned state stay intact.

No SSH/submission. CPU preparation makes fresh, audited runtime-only checkpoint
copies. Actual five-GPU continuation/resume and host-memory admission precede
formal training. Frozen probes/segmentation retain the original GT protocol.
"""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from random_depth_study import TREATMENTS, ROOT, load_config, require_allocation, treatment_plan

TRAINING_COMMIT = 'a31f9039f9fe3e265a503a1ffdedc03b71e361a0'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def commit(root):
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def settings(path):
    cfg = load_config(path)
    recovery = cfg['recovery']
    if (cfg['node'] != 'eowyn' or cfg['workers_per_rank'] != 0 or cfg['gpus_per_experiment'] != 5
            or recovery['expected_updates'] != dict(zip(TREATMENTS, (2250,3000)))
            or commit(recovery['training_root']) != TRAINING_COMMIT):
        raise ValueError('Use the matched worker-free five-GPU recovery with the original training code')
    for key in ('training_root','parent_run_root'):
        if not Path(recovery[key]).is_absolute():
            raise ValueError('Explicit recovery paths required')
    if Path(cfg['run_root']) == Path(recovery['parent_run_root']):
        raise ValueError('Preserve old failed outputs; recover into a fresh root')
    if subprocess.check_output(['git','-C',recovery['training_root'],'status','--porcelain'],text=True).strip():
        raise ValueError('Original immutable training source must remain clean')
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0] != cfg['node']:
        raise ValueError('A Slurm allocation on the existing eowyn data node is required')
    return cfg


def old_command(cfg, treatment, readout=None):
    plan = treatment_plan(cfg, treatment)
    task = plan['pretrain'] if readout is None else next(x for x in plan['downstream'] if x['name']==readout)
    old = str(Path(cfg['recovery']['training_root']))
    return [old+x[len(str(ROOT)):] if x.startswith(str(ROOT)+'/') else x for x in task['argv']]


def memory_group():
    """Use the lowest finite memory limit in OUR cgroup ancestry; no admin access."""
    candidates=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if controllers=='' or 'memory' in controllers.split(','):
            base=Path('/sys/fs/cgroup') if controllers=='' else Path('/sys/fs/cgroup/memory')
            path=base/relative.lstrip('/')
            while path.is_relative_to(base):
                names=('memory.current','memory.peak','memory.max') if controllers=='' else (
                    'memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes')
                files=[path/n for n in names]
                if files[0].exists() and files[2].exists():
                    if not files[1].exists(): files[1]=files[0]
                    raw=files[2].read_text().strip()
                    if raw.isdigit() and 0<int(raw)<2**60:
                        candidates.append((int(raw),files))
                if path==base: break
                path=path.parent
    if not candidates:
        raise RuntimeError('Cannot verify the actual Slurm host-memory limit')
    return min(candidates,key=lambda x:x[0])[1]


def last_update(directory, default):
    path=directory/'metrics.jsonl'
    if not path.exists(): return default
    with path.open('rb') as handle:
        handle.seek(0,2)
        handle.seek(max(0,handle.tell()-16384))
        for line in reversed(handle.read().splitlines()):
            try: return json.loads(line)['update']
            except (ValueError,KeyError): pass
    return default


def run_child(cfg, treatment, stop=0, readout=None):
    command=old_command(cfg,treatment,readout)
    directory=Path(command[command.index('--run-dir')+1])
    if readout is None:
        command += ['--resume']
        if stop: command += ['--stop-after-updates',str(stop)]
    elif (directory/'config.json').exists():
        raise FileExistsError('Readout already exists; do not duplicate it')
    old=Path(cfg['recovery']['training_root'])
    environment=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',TORCH_HOME=str(old/'.cache/torch'),MPLCONFIGDIR=str(old/'.cache/matplotlib'))
    files=memory_group()
    limit=int(files[2].read_text())
    start_update=last_update(directory,cfg['recovery']['expected_updates'][treatment])
    directory.mkdir(parents=True,exist_ok=True)
    child=None
    previous={}
    def stop_child(signum,frame):
        if child is not None:
            if readout is None:
                (directory/'REQUEST_CHECKPOINT').touch()
            else:
                child.send_signal(signal.SIGUSR1)
    if readout is None:
        (directory/'REQUEST_CHECKPOINT').unlink(missing_ok=True)
    for signum in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        previous[signum]=signal.signal(signum,stop_child)
    # Use the original wrapper for DDP signal handling/output locks/audit gates.
    if readout is None:
        command=['bash',str(old/'cluster_rpl/random_depth_job.sh'),str(cfg['_config_path']),
                 treatment,'--resume']+(['--stop-after-updates',str(stop)] if stop else [])
        environment['DD2414_PROJECT_ROOT']=str(old)
    samples=[]
    started=time.monotonic()
    log=directory/'recovery_resources.jsonl'
    try:
        child=subprocess.Popen(command,cwd=old,env=environment)
        last_print=started
        while child.poll() is None:
            update=last_update(directory,start_update)
            row=dict(seconds=time.monotonic()-started,update=update,
                host_current_gib=int(files[0].read_text())/2**30,
                host_peak_gib=int(files[1].read_text())/2**30,host_limit_gib=limit/2**30,gpus=[])
            gpu_ids=os.environ.get('SLURM_JOB_GPUS','')
            if gpu_ids and all(part.isdigit() for part in gpu_ids.split(',')):
                try:
                    result=subprocess.check_output(['nvidia-smi','--id='+gpu_ids,
                        '--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                        text=True,timeout=3)
                    row['gpus']=[dict(index=int(x[0]),utilization=float(x[1]),used_mib=float(x[2]),total_mib=float(x[3]))
                        for x in csv.reader(result.splitlines())]
                except (subprocess.SubprocessError,ValueError): pass
            with log.open('a') as handle: handle.write(json.dumps(row)+'\n')
            samples.append(row)
            if time.monotonic()-last_print>30:
                print('RECOVERY_MEMORY '+json.dumps(row),flush=True)
                last_print=time.monotonic()
            time.sleep(2)
        code=child.wait()
    finally:
        for signum,handler in previous.items(): signal.signal(signum,handler)
    summary=read(directory/'summary.json') if (directory/'summary.json').exists() else {}
    if code==0 and summary.get('status')!='complete': code=75
    steady=[x for x in samples if x['update']>=start_update+5 and x['gpus']]
    gpu_summary=[]
    for index in sorted({gpu['index'] for row in steady for gpu in row['gpus']}):
        rows=[gpu for row in steady for gpu in row['gpus'] if gpu['index']==index]
        gpu_summary.append(dict(index=index,samples=len(rows),
            mean_utilization=sum(x['utilization'] for x in rows)/len(rows),
            min_utilization=min(x['utilization'] for x in rows),max_utilization=max(x['utilization'] for x in rows),
            maximum_used_mib=max(x['used_mib'] for x in rows),total_mib=rows[0]['total_mib']))
    telemetry=dict(exit_code=code,elapsed_seconds=time.monotonic()-started,
        host_peak_gib=max([x['host_peak_gib'] for x in samples],default=0),host_limit_gib=limit/2**30,
        start_update=start_update,end_update=last_update(directory,start_update),
        steady_gpu_samples=len(steady),gpu_summary=gpu_summary,
        host_peak_counter_supported=files[0]!=files[1],
        scope='Sampled GPU activity and cgroup RAM; GPU utilization is not peak FLOP efficiency')
    if len(steady)>1 and steady[-1]['seconds']>steady[0]['seconds']:
        telemetry['measured_training_samples_per_second'] = (
            5*(steady[-1]['update']-steady[0]['update'])/(steady[-1]['seconds']-steady[0]['seconds']))
        telemetry['measured_window_seconds']=steady[-1]['seconds']-steady[0]['seconds']
    write(directory/'recovery_resource_summary.json',telemetry)
    return code,telemetry


def inspect(cfg,treatment,expected):
    import torch
    directory=Path(cfg['run_root'])/treatment/'pretrain'
    value=torch.load(directory/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    if expected is None: expected=value['progress']['update']
    if value['progress']['update']!=expected or value['progress']['samples_seen']!=expected*5:
        raise ValueError('Unexpected resumed progress')
    if not cfg['recovery']['expected_updates'][treatment]<=expected<=281300:
        raise ValueError('Recovery cursor lies outside the inherited total budget')
    if (value['run_config']['num_workers']!=0 or value['world_size']!=5
            or value['run_config']['epochs']!=50 or len(value['distributed_state'])!=5):
        raise ValueError('Recovery altered the scientific budget or loader recipe')
    for state in value['distributed_state']:
        if state['sampler']['position']!=value['sampler']['position']:
            raise ValueError('Rank cursor disagreement')
    if not all(not torch.is_floating_point(x) or bool(torch.isfinite(x).all()) for x in value['model'].values()):
        raise ValueError('Non-finite model state')
    for state in value['optimizer']['state'].values():
        if not all(not isinstance(x,torch.Tensor) or not torch.is_floating_point(x) or bool(torch.isfinite(x).all())
                   for x in state.values()):
            raise ValueError('Non-finite optimizer state')
    return value


def require_admission(cfg):
    data=read(Path(cfg['run_root'])/'worker_free_recovery_passed.json')
    public={k:v for k,v in cfg.items() if not k.startswith('_')}
    if (data.get('status')!='passed' or data['settings']!=public or data['controller_commit']!=commit(ROOT)
            or set(data['treatments'])!=set(TREATMENTS)):
        raise ValueError('Both exact-config worker-free CUDA/host-memory resume admissions are required')
    for treatment in TREATMENTS:
        result=data['treatments'][treatment]
        parent=cfg['recovery']['expected_updates'][treatment]
        start=result.get('pilot_initial_update',-1)
        if (result.get('status')!='passed' or result.get('parent_update')!=parent
                or start<parent or result.get('resumed_update')!=start+101):
            raise ValueError('Incomplete/non-matching recovery admission')
        for name in ('initial_segment','fresh_process_segment'):
            segment=result.get(name,{})
            if (segment.get('exit_code')!=75 or segment.get('host_limit_gib',0)<=0
                    or not 0<segment.get('host_peak_gib',0)<.9*segment['host_limit_gib']):
                raise ValueError('Missing actual finite-update/host-memory admission')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','pilot','pretrain','readout'],required=True)
    parser.add_argument('--treatment',choices=TREATMENTS)
    parser.add_argument('--readout',choices=['frozen_camera_bev_probe','frozen_fused_bev_probe','end_to_end_segmentation'])
    args=parser.parse_args()
    cfg=settings(args.config)
    public=cfg.copy()
    cfg['_config_path']=str(args.config.resolve())
    if args.phase=='prepare':
        from recovery_checkpoint import migrate
        reports={}
        for treatment in TREATMENTS:
            reports[treatment]=migrate(Path(cfg['recovery']['parent_run_root'])/treatment/'pretrain/checkpoints/latest.pt',
                Path(cfg['run_root'])/treatment/'pretrain',cfg['recovery']['training_root'],
                cfg['recovery']['expected_updates'][treatment],treatment)
            gc.collect()
        write(Path(cfg['run_root'])/'runtime_migrations.json',dict(settings=public,treatments=reports))
        return
    if args.phase=='readout':
        if not args.treatment or not args.readout: parser.error('Explicit treatment/readout required')
        require_admission(cfg)
        code,_=run_child(cfg,args.treatment,readout=args.readout)
        raise SystemExit(code)
    require_allocation(cfg,os.environ,socket.gethostname())
    if args.phase=='pretrain':
        if not args.treatment: parser.error('Explicit treatment required')
        require_admission(cfg)
        code,_=run_child(cfg,args.treatment)
        raise SystemExit(code)
    results={}
    for treatment in TREATMENTS:
        origin=cfg['recovery']['expected_updates'][treatment]
        before=inspect(cfg,treatment,None)
        start=before['progress']['update']
        directory=Path(cfg['run_root'])/treatment/'pretrain'
        if last_update(directory,origin)!=start:
            raise ValueError('Archive/reconcile uncheckpointed log tails before a recovery retry')
        del before
        code,first=run_child(cfg,treatment,stop=start+100)
        if code!=75: raise RuntimeError('First recovery segment failed: '+treatment)
        before=inspect(cfg,treatment,start+100)
        code,second=run_child(cfg,treatment,stop=start+101)
        if code!=75: raise RuntimeError('Fresh-process recovery segment failed: '+treatment)
        after=inspect(cfg,treatment,start+101)
        rows=[json.loads(x) for x in (Path(cfg['run_root'])/treatment/'pretrain/metrics.jsonl').read_text().splitlines()]
        if [x['update'] for x in rows]!=list(range(origin+1,start+102)):
            raise ValueError('Recovery replayed/skipped a logged update')
        if not all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']) for x in rows):
            raise ValueError('Non-finite recovery loss/gradient')
        if not any(not __import__('torch').equal(x,after['model'][k]) for k,x in before['model'].items()):
            raise ValueError('No learned state changed on resume')
        if max(first['host_peak_gib'],second['host_peak_gib'])>=.9*first['host_limit_gib']:
            raise RuntimeError('Insufficient verified host-memory headroom')
        results[treatment]=dict(status='passed',parent_update=origin,pilot_initial_update=start,resumed_update=start+101,
            initial_segment=first,fresh_process_segment=second)
        print('RECOVERY_TREATMENT_PASSED '+json.dumps(dict(treatment=treatment,**results[treatment])),flush=True)
        del before,after
        gc.collect()
    write(Path(cfg['run_root'])/'worker_free_recovery_passed.json',dict(status='passed',settings=public,
        controller_commit=commit(ROOT),training_code_commit=TRAINING_COMMIT,treatments=results))
    print('BOTH_WORKER_FREE_RECOVERY_PILOTS_PASSED',flush=True)


if __name__=='__main__': main()
