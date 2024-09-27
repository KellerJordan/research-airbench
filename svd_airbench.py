"""
svd_airbench.py

Variant of clean_airbench which uses a slow SVD-based optimizer.

Attains 94.00(n=80) accuracy in only 8 epochs.
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

w = 1.0
hyp = {
    'opt': {
        'epochs': 8,
        'batch_size': 1000,
        'lr': 10.0,             # learning rate per 1024 examples -- 5.0 is optimal with no smoothing, 10.0 with smoothing.
        'filter_lr': 0.20,      # the spectral norm of the rotation matrix added each step
        'momentum': 0.85,
        'weight_decay': 0.015,  # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,    # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': int(64*w),
            'block2': int(256*w),
            'block3': int(256*w),
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

        shape = [len(param)]+[1]*(len(param.shape)-1)
        # normalize each filter
        filter_data_norms = param.data.reshape(len(param), -1).norm(dim=1)
        param.data.div_(filter_data_norms.view(*shape))
        ## normalize each filter gradient
        #filter_grad_norms = d_p.reshape(len(d_p), -1).norm(dim=1)
        #update = d_p / filter_grad_norms.view(*shape)
        # whiten the gradient
        #U, _, V = d_p.reshape(len(d_p), -1).float().svd()
        U, _, VT = torch.linalg.svd(d_p.reshape(len(d_p), -1).float(), full_matrices=False)
        update = (U @ VT).to(param.dtype).view(param.shape)
        # take a step using the normalized gradients
        param.data.add_(update, alpha=-lr)

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
    def __init__(self, num_features, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps)
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
#          Training and Inference          #
############################################

def train(train_loader):

    momentum = hyp['opt']['momentum']
    epochs = hyp['opt']['epochs']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * train_loader.batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    total_train_steps = epochs * len(train_loader)

    model = make_net()

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if len(p.shape) < 4 and p.requires_grad and 'norm' in n]
    other_params = [p for n, p in model.named_parameters() if len(p.shape) < 4 and p.requires_grad and 'norm' not in n]
    optimizer1 = RenormSGD(filter_params, lr=hyp['opt']['filter_lr'])
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
    def get_lr(step):
        return 1 - step / total_train_steps
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr)

    train_loader.epoch = 0
    from tqdm import tqdm
    for epoch in tqdm(range(epochs)):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

    return model

if __name__ == '__main__':

    import sys
    with open(sys.argv[0]) as f:
        code = f.read()

    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'], altflip=True)
    test_loader = CifarLoader('/tmp/cifar10', train=False)

    #print(evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level']))
    accs = torch.tensor([evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level']) for _ in range(3)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))
    import os
    import uuid
    os.makedirs('logs', exist_ok=True)
    log = {'code': code, 'accs': accs}
    torch.save(log, 'logs/%s.pt' % uuid.uuid4())

