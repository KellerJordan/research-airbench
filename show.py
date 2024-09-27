import os
import glob
import torch
pp = sorted(glob.glob('logs/*.pt'), key=lambda x: os.path.getmtime(x))
objs = [torch.load(p) for p in pp]
assert len(objs) % 8 == 0
accs = [o['accs'].mean() for o in objs]
print(torch.tensor(accs).reshape(-1, 8).mean(1))
