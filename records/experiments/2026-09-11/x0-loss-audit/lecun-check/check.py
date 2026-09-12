import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

root=Path('/home/yspark/projects/DexPrompt/out/student-x0-low-noise-code-v1')
sys.path.insert(0,str(root/'scripts'))
import train_hx5_student as trainer

def forbidden(*args,**kwargs):
    raise RuntimeError('training forbidden in this check')
torch.Tensor.backward=forbidden
torch.autograd.backward=forbidden
for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):
    cls.step=forbidden
config=json.loads((root/'configs/hx5_student.single-bottle-x0-low-noise.lecun.json').read_text())
base=json.loads((root/'configs/hx5_student.single-bottle-x0.lecun.json').read_text())
comparison=dict(config);assert comparison.pop('low_noise_loss')=={'steps':10,'relative_weight':4.0}
assert comparison==base
out=Path('/data1/yspark/projects/dexprompt/student-checks/x0-low-noise-preflight-v1')
args=SimpleNamespace(data=Path('/home/yspark/projects/DexPrompt/out/student-single-bottle-v1'),cache=Path('/home/yspark/projects/DexPrompt/out/student-single-bottle-v1-cache'),config=None,out=out/'preflight',allow_train_only_diagnostic=True,device='cuda:0',full_model=True,sample_index=[0,200])
report=trainer.inspect(args,config)
assert report['training_ready'],report
args.out=out/'dry-run'
trainer.dry_run(args,config)
print('CHECK_COMPLETED_NO_TRAINING',flush=True)
