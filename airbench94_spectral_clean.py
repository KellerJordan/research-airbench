"""
airbench94_spectral_clean.py
Should(?) attain 94.04 mean accuracy (n=200 trials)
* Removes compilation and freezing from the speedrun code.
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import airbench

torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'train_epochs': 8,
        'batch_size': 2000,
        'lr': 6.5,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.015,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#           Spectral SGD-momentum           #
#############################################

@torch.compile
def zeroth_power_via_newton(G, steps=9):
    """
    Computing zeroth matrix powers via Lakic 1998.
    paper: "On the Computation of the Matrix k-th Root"
    Suppose we have a matrix G = USV^T and we want to compute G^0 defined via G^0 = UV^T.
    We might want to do this to run "stochastic spectral descent" of Carlson et al 2015.
    The naive way to do this is via the SVD. But we can also just do (GG^T)^(-1/2) G or
    alternatively G (G^TG)^(-1/2) and apply the iterative method from Lakic 1998.
    In particular, we implement the first special case of Alg 1 in that paper.

    Code taken from: https://gist.github.com/jxbz/fe235ee1c72b8b41ccd0d02b43378cf2
    https://x.com/jxbz/status/1821610280708948103
    Modifications: To speed things up, I am running this in bfloat16 using torch.compile.
    """

    orig_dtype = G.dtype
    G = G.bfloat16()

    d1, d2 = G.shape
    d = min(d1, d2)
    I = torch.eye(d, device=G.device, dtype=G.dtype)

    # store the smaller of the squares as S
    S = G @ G.T if d1 < d2 else G.T @ G
    S_norm = torch.linalg.matrix_norm(S, ord='fro') # there is freedom here. See Lakic (1998) Thm 2.3
    S /= S_norm

    # Now let's set up the state for the Lakic (1998) method
    N = S
    X = I.clone()

    # Now let's run the iteration
    for step in range(steps):
        U = (3 * I - N) / 2
        X = X @ U if step > 0 else U # optimization since X = I on step 0
        if step < steps-1: # optimization suggested by @EitanTurok
            N = N @ U @ U
    X /= S_norm.sqrt()

    # X should now store either (G G^T)^(-1/2) or (G^T G)^(-1/2)
    O = X @ G if d1 < d2 else G @ X
    return O.to(orig_dtype)

class SpectralSGDM(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if momentum != 0:
                    if 'momentum_buffer' not in state.keys():
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeroth_power_via_newton(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

## The eigenvectors of the covariance matrix of 2x2 patches of CIFAR-10 divided by sqrt eigenvalues.
eigenvectors_scaled = torch.tensor([
 8.9172e-02,  8.9172e-02,  8.8684e-02,  8.8623e-02,  9.3872e-02,  9.3872e-02,  9.3018e-02,  9.3018e-02,
 9.0027e-02,  9.0027e-02,  8.9233e-02,  8.9172e-02,  3.8818e-01,  3.8794e-01,  3.9111e-01,  3.9038e-01,
-1.0767e-03, -1.3609e-03,  3.8567e-03,  3.2330e-03, -3.9087e-01, -3.9087e-01, -3.8428e-01, -3.8452e-01,
-4.8242e-01, -4.7485e-01,  4.5435e-01,  4.6216e-01, -4.6240e-01, -4.5557e-01,  4.8975e-01,  4.9658e-01,
-4.3311e-01, -4.2725e-01,  4.2285e-01,  4.2896e-01, -5.0781e-01,  5.1514e-01, -5.1562e-01,  5.0879e-01,
-5.1807e-01,  5.2783e-01, -5.2539e-01,  5.1904e-01, -4.6460e-01,  4.7070e-01, -4.7168e-01,  4.6240e-01,
-4.7290e-01, -4.7461e-01, -5.0635e-01, -5.0684e-01,  9.5410e-01,  9.5117e-01,  9.2090e-01,  9.1846e-01,
-4.7363e-01, -4.7607e-01, -5.0439e-01, -5.0586e-01, -1.2539e+00,  1.2490e+00,  1.2383e+00, -1.2354e+00,
-1.2637e+00,  1.2666e+00,  1.2715e+00, -1.2725e+00, -1.1396e+00,  1.1416e+00,  1.1494e+00, -1.1514e+00,
-2.8262e+00, -2.7578e+00,  2.7617e+00,  2.8438e+00,  3.9404e-01,  3.7622e-01, -3.8330e-01, -3.9502e-01,
 2.6602e+00,  2.5801e+00, -2.6055e+00, -2.6738e+00, -2.9473e+00,  3.0312e+00, -3.0488e+00,  2.9648e+00,
 3.9111e-01, -4.0063e-01,  3.7939e-01, -3.7451e-01,  2.8242e+00, -2.9023e+00,  2.8789e+00, -2.8008e+00,
 2.6582e+00,  2.3105e+00, -2.3105e+00, -2.6484e+00, -5.9336e+00, -5.1680e+00,  5.1719e+00,  5.9258e+00,
 3.6855e+00,  3.2285e+00, -3.2148e+00, -3.6992e+00, -2.4668e+00,  2.8281e+00, -2.8379e+00,  2.4785e+00,
 5.4062e+00, -6.2031e+00,  6.1797e+00, -5.3906e+00, -3.3223e+00,  3.8164e+00, -3.8223e+00,  3.3340e+00,
-8.0000e+00,  8.0000e+00,  8.0000e+00, -8.0078e+00,  9.7656e-01, -9.9414e-01, -9.8584e-01,  1.0039e+00,
 7.5938e+00, -7.5820e+00, -7.6133e+00,  7.6016e+00,  5.5508e+00, -5.5430e+00, -5.5430e+00,  5.5352e+00,
-1.2133e+01,  1.2133e+01,  1.2148e+01, -1.2148e+01,  7.4141e+00, -7.4180e+00, -7.4219e+00,  7.4297e+00,
]).reshape(12, 3, 2, 2)

def make_net():
    widths = hyp['net']['widths']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1']),
        ConvGroup(widths['block1'], widths['block2']),
        ConvGroup(widths['block2'], widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

############################################
#                Training                  #
############################################

def main(train_ooader):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    total_train_steps = ceil(len(train_loader) * epochs)

    current_steps = 0
    model = make_net()

    # Create optimizers for train whiten bias stage
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=[model[0].bias, model[-2].weight], lr=lr, weight_decay=wd/lr)]
    optimizer1 = SpectralSGDM(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
    def get_lr(step):
        return 1 - step / total_train_steps
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr)

    for epoch in range(ceil(epochs)):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            current_steps += 1
            if current_steps >= total_train_steps:
                break
    
    return model
    
train_loader = airbench.CifarLoader('cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'])
test_loader = airbench.CifarLoader('cifar10', train=False, batch_size=2000)
from tqdm import tqdm
accs = []
for _ in tqdm(range(10)):
    acc = airbench.evaluate(main(train_loader), test_loader, tta_level=2)
    print(acc)
    accs.append(acc)
print(torch.tensor(accs).mean())
#print(airbench.evaluate(model, test_loader, tta_level=2))
