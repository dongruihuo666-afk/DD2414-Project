"""Compare a preserved checkout and package in separate CPU-only processes.

This exercises production functions, model state names, gradients and resumed
optimizer behavior. The reference checkout is never edited or executed as a
training job. Run with --implementation legacy/package and --source-root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch import nn

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--implementation',choices=['legacy','package'],required=True)
parser.add_argument('--source-root',type=Path,required=True)
args=parser.parse_args()
sys.path[:0]=[str(args.source_root/'scripts'),str(args.source_root)]
if args.implementation=='package':
    sys.path.insert(0,str(args.source_root/'src'))
    from dd2414.losses.semantic import semantic_loss
    from dd2414.objectives.radar_motion import motion_loss
    from dd2414.models.multisweep import MultiSweepTeacher,configure
    from dd2414.models.metric_bev import DepthProbabilityVox,project_teacher_depth,projected_depth_loss
    from dd2414.runtime.fullsize import AcknowledgedShuffleSampler,learning_rate_at_step
    from dd2414.runtime.checkpoints import save_training_checkpoint,load_training_checkpoint
else:
    from semantic_distill_smoke import semantic_loss
    from compare_motion_target_bevcar_tiny import motion_loss
    from multisweep_teacher import MultiSweepTeacher,configure
    from metric_depth_bev import DepthProbabilityVox,project_teacher_depth,projected_depth_loss
    from fullsize_training import AcknowledgedShuffleSampler,learning_rate_at_step
    from training_checkpoint import save_training_checkpoint,load_training_checkpoint
import utils.vox
from nets.radar_encoder import transform_radar_to_camera_bev

torch.set_num_threads(1)
random.seed(19);np.random.seed(19);torch.manual_seed(19)
def tensor(value):
    x=value.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
result={}
prediction=torch.randn(2,7,4,5,requires_grad=True)
loss,cosine=semantic_loss(prediction,torch.randn_like(prediction),torch.rand(2,4,5))
loss.backward();result['semantic']=dict(loss=tensor(loss),cosine=tensor(cosine),gradient=tensor(prediction.grad))
prediction=torch.randn(1,2,4,5,requires_grad=True)
loss,count=motion_loss(prediction,torch.randn(2,4,5),torch.ones(4,5))
loss.backward();result['motion']=dict(loss=tensor(loss),count=count,gradient=tensor(prediction.grad))
empty=torch.randn(1,2,4,5,requires_grad=True)
loss,count=motion_loss(empty,torch.zeros(2,4,5),torch.zeros(4,5));loss.backward()
result['empty_motion']=dict(loss=tensor(loss),count=count,gradient=tensor(empty.grad))

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,128,1),nn.BatchNorm2d(128))
        self.bevcar_encoder=nn.Conv2d(128,128,1)
        self.bev_compressor=nn.Conv2d(128,128,1)
        self.decoder=nn.Conv2d(128,1,1)
    def forward(self,rgb_camXs,**kwargs):
        image=self.encoder(rgb_camXs)
        return self.bev_compressor(self.bevcar_encoder(image)),image,image
model=MultiSweepTeacher(Student());configure(model,'pretrain')
shared,image,_=model.features((torch.randn(2,3,4,5),None,None,None,None))
objective=model.bev_head(shared).square().mean()+model.image_head(image).square().mean()
objective.backward()
result['model']=dict(state={k:tensor(v) for k,v in model.state_dict().items()},
    gradients={k:None if v.grad is None else tensor(v.grad) for k,v in model.named_parameters()},output=tensor(shared))

features=torch.randn(6,4,5,6,requires_grad=True);features.data[:,0]=0.
head=nn.Conv2d(4,80,1)
base=object.__new__(utils.vox.Vox_util)
cameras=torch.eye(4).repeat(6,1,1);cameras[:,2,3]=torch.arange(6).float()*.5
xyz=torch.linspace(-2,90,6)[None,:,None].expand(6,-1,3).clone()
wrapper=DepthProbabilityVox(base,head,True)
output=wrapper.reduce_camera_features(features,cameras,cameras,3,1,2,batch_size=2,camera_count=3,xyz_camA=xyz)
g=wrapper.geometry
target=project_teacher_depth(torch.full((6,1,5,6),30.),g['grid'],g['valid'],g['feature_size'])
loss=output.square().sum()+.05*projected_depth_loss(g['log_probability_bev'],target)
loss.backward()
result['depth']=dict(output=tensor(output),grid=tensor(g['grid']),target=tensor(target),
    loss=tensor(loss),features_gradient=tensor(features.grad),head_gradient=tensor(head.weight.grad))

radar=torch.zeros(1,3,19);radar[0,:,:3]=torch.tensor([[1.,2.,3.],[-1.,5.,2.],[0.,0.,0.]])
radar[0,:,6:10]=torch.tensor([[1.,2.,3.,4.],[-3.,4.,5.,-2.],[0.,0.,0.,0.]])
transform=torch.eye(4)[None];transform[0,:3,:3]=torch.tensor([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
result['radar_rotation']=tensor(transform_radar_to_camera_bev(radar,transform))
sampler=AcknowledgedShuffleSampler(list(range(10)),125);iterator=iter(sampler)
requested=[next(iterator) for _ in range(6)];sampler.acknowledge(4)
clone=AcknowledgedShuffleSampler(list(range(10)),125);clone.load_state_dict(sampler.state_dict())
result['sampler']=dict(requested=requested,state=sampler.state_dict(),remaining=list(clone))
result['schedule']=[learning_rate_at_step(.0002,i,100,10,.1) for i in [0,9,50,99]]

class Scaler:
    def state_dict(self):return {}
    def load_state_dict(self,value):assert value=={}
net=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Linear(4,1))
optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
optimizer.zero_grad();net(torch.randn(3,3)).square().mean().backward();optimizer.step()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'state.pt';config=dict(experiment='parity',epochs=2)
    save_training_checkpoint(path,model=net,optimizer=optimizer,scaler=Scaler(),sampler=sampler,
        progress=dict(epoch=0,update=1),run_config=config)
    restored=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Linear(4,1))
    opt=torch.optim.AdamW(restored.parameters(),lr=.001)
    other=AcknowledgedShuffleSampler(list(range(10)),125)
    progress=load_training_checkpoint(path,model=restored,optimizer=opt,scaler=Scaler(),sampler=other,
        expected_run_config=config,map_location='cpu')
    opt.zero_grad();restored(torch.randn(3,3)).square().mean().backward();opt.step()
    result['resume']=dict(progress=progress,sampler=other.state_dict(),
        model={k:tensor(v) for k,v in restored.state_dict().items()},
        optimizer={str(k):{n:tensor(v) if isinstance(v,torch.Tensor) else v for n,v in row.items()} for k,row in opt.state_dict()['state'].items()})
print(json.dumps(result,sort_keys=True))
