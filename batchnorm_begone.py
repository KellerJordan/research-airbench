"""
Things which didn't help:
* Doing full-parameter normalization instead of filter-wise (this was much less stable)
* Using gamma/beta affine scale/shift coefficients after the convs, like NFNet
* Using 0.85 instead of 0.9 momentum (with batch size set to 500 as it is)
* Turning off label smoothing (reduces from 92.9 to 92.5)
* Warming up from zero instead of from 0.2 (93.0 -> 92.9, not quite stat sig)
* Turning off tta from 2 to 0 (93.7 -> 93.0)
Result: at lr=0.03, we got 93.69 in n=50.
Result: now with both non-filters at lr=0.01, got 93.70 in n=50.
* Blows up with non-filters at lr=0.02.
Result: now with both non-filters at lr=0.005, got 93.76 in n=50.
Result: now projecting out the parallel direction, got 93.74 in n=50.

Learning rate for filters
* lr=0.03 -> 93.74(50)
* lr=0.04 -> 93.57(50)

Weight decay on the non-filters doesn't help
* wd=0.0 -> 93.74(50)
* wd=0.05 -> 93.71(10)
* wd=0.1 -> 93.73(10)
* wd=0.2 -> 93.61(10)
* wd=0.5 -> 93.51(10)

Now bs=1000
* lr=0.03 -> 93.07(10)
* lr=0.04 -> 92.96(10)
* lr=0.03, non-filter lr=5/bs -> garbage
* momentum=0.85 -> 92.58(10)

Now bs=1000 epochs=30
* lr=0.03 -> 93.97(10)
* lr=0.04 -> 94.16(10)
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from airbench import evaluate, CifarLoader

torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'epochs': 15,
        'momentum': 0.9,
        'batch_size': 500,
        'label_smoothing': 0.2,
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
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}

#############################################
#          Renormalized Optimizer           #
#############################################

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

class RenormSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            d_p_list = [p.grad for p in params_with_grad]
            momentum_buffer_list = [self.state[p].get('momentum_buffer') for p in params_with_grad]

            renorm_sgd(params_with_grad,
                       d_p_list,
                       momentum_buffer_list,
                       momentum=group['momentum'],
                       lr=group['lr'],
                       dampening=group['dampening'],
                       nesterov=group['nesterov'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                self.state[p]['momentum_buffer'] = momentum_buffer

        return loss

def renorm_sgd(params: List[Tensor],
               d_p_list: List[Tensor],
               momentum_buffer_list: List[Optional[Tensor]],
               *,
               momentum: float,
               lr: float,
               dampening: float,
               nesterov: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        g = d_p
        shape = [len(g)]+[1]*(len(g.shape)-1)
        # project out the parallel component
        dot = (g * param.data).reshape(len(g), -1).sum(1)
        g = g - dot.view(*shape) * param.data
        # normalize each filter's gradient
        grad_scale = g.reshape(len(g), -1).norm(dim=1)
        g = g / grad_scale.view(*shape)
        # take a step
        param.data.add_(g, alpha=-lr)
        # re-normalize each filter
        norm_scale = param.data.reshape(len(param), -1).norm(dim=1)
        param.data.div_(norm_scale.view(*shape))

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

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
        w[w.size(1):] *= 3**0.5

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = Conv(channels_out, channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activ(x)
        x = self.conv2(x)
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
    return net

#############################################
#          Training and Evaluation          #
#############################################

def train(train_loader):

    epochs = hyp['opt']['epochs']

    model = make_net()
    total_train_steps = epochs * len(train_loader)

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    other_params = [p for p in model.parameters() if len(p.shape) < 4 and p.requires_grad]
    optimizer1 = RenormSGD(filter_params, lr=0.03, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer2 = torch.optim.SGD(other_params, lr=2.5 / hyp['opt']['batch_size'],
                                 momentum=hyp['opt']['momentum'], nesterov=True)
    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.2)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return (1 - frac)
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr)

    current_steps = 0
    train_loader.epoch = 0
    model.train()
    from tqdm import tqdm
    for epoch in tqdm(range(epochs)):
        for inputs, labels in train_loader:

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none',
                                   label_smoothing=hyp['opt']['label_smoothing']).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            current_steps += 1
            if current_steps == total_train_steps:
                break

    return model

if __name__ == '__main__':
    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=hyp['opt']['batch_size'],
                               aug=hyp['aug'], altflip=True)
    test_loader = CifarLoader('/tmp/cifar10', train=False)

    print(evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level']))
    print(torch.std_mean(torch.tensor([evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level'])
                                       for _ in range(50)])))

