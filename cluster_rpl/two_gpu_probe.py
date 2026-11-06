"""Synthetic CUDA/NCCL/DDP probe; run only in an authorized two-GPU Slurm job.

No dataset, teacher, project checkpoint, or full research model is loaded.
Use torchrun with two processes and an explicit short Slurm wall limit.
"""
import datetime
import json
import os
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A specifically authorized two-GPU Slurm allocation is required')
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2
    assert torch.cuda.device_count() == 2, 'Expected exactly two Slurm-visible GPUs'
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    props = torch.cuda.get_device_properties(rank)
    assert props.major >= 8 and torch.cuda.is_bf16_supported(), 'Native BF16 required'
    print(json.dumps(dict(event='cuda_visible', hostname=socket.gethostname(),
        job_id=os.environ.get('SLURM_JOB_ID'), rank=rank,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=props.name,
        uuid=str(props.uuid), memory_gib=round(props.total_memory/2**30,2),
        torch=torch.__version__, cuda_runtime=torch.version.cuda)), flush=True)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=60))
    try:
        value = torch.tensor([rank+1.0], device='cuda')
        dist.all_reduce(value)
        assert value.item() == 3.0, 'NCCL all-reduce failed'
        torch.manual_seed(125+rank)
        model = DistributedDataParallel(torch.nn.Linear(64,32).cuda(), device_ids=[rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        inputs = torch.randn(8,64,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss = model(inputs).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        fingerprint = torch.stack([p.detach().float().sum() for p in model.parameters()])
        fingerprints = [torch.zeros_like(fingerprint) for _ in range(2)]
        dist.all_gather(fingerprints, fingerprint)
        assert torch.allclose(fingerprints[0], fingerprints[1], atol=1e-6, rtol=1e-6), 'DDP parameters differ'
        torch.cuda.synchronize()
        print(json.dumps(dict(event='rank_passed', rank=rank, nccl_sum=value.item(),
            finite_loss=loss.item(), ddp_parameters_synchronized=True,
            scope='Synthetic CUDA/NCCL/DDP compatibility only; not full-model training or policy authorization')), flush=True)
        dist.barrier()
        if rank == 0:
            print('TWO_GPU_PROBE_PASSED',flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
