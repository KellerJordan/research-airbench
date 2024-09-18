"""
casted2_airbench.py

Variant of clean_airbench which uses the renormalized optimizer.
Also removes dirac initialization, for the purpose of casting experiments.

In full precision, attains 93.78 mean accuracy (n=50).
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
        'epochs': 10,
        'batch_size': 1000,
        'lr': 10.0,             # learning rate per 1024 examples -- 5.0 is optimal with no smoothing, 10.0 with smoothing.
        'filter_lr': 0.07,      # the norm of the orthogonal update applied to each conv filter each step, which are all norm-1
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
        'width_factor': 1,
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'scaling_factor': 1/9,
        'tta_level': 2,

        # M=(number of mantissa bits), E=(number of exponent bits), A=(log_2 of smallest normal number)
        # For example, (7, 0, 7) is int8 (everything subnormal), and (2, 5, -14) is torch.float8_e5m2.
        # And (0, 1, 0) is ternary. Among positives, it represents 1 as a normal number, and 0 as the subnormal number.
        # Note that there's always a sign bit, so ternary is {-1, 0, 1}.
        # See `cast_tensor` for more documentation.
        'MEA': (10, 5, -14), # torch.half -> 93.78(n=50)
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
        # normalize each filter gradient
        filter_grad_norms = d_p.reshape(len(d_p), -1).norm(dim=1)
        update = d_p / filter_grad_norms.view(*shape)
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

def cast_tensor(x, M, E, A):
    """
    Rounds every value in the tensor x to the nearest representable floating point number.
    Where the floating point representation has M mantissa bits, smallest exponent A, and E exponent bits.

    Therefore (only considering positive numbers):
    * The subnormal numbers will be {0, 2**-M * 2**A), 2 * 2**-M * 2**A, ..., (2**M - 1) * 2**-M * 2**A}
    * The smallest denormal number will be 2**A
    * The largest denormal number will be 2**(A+2**E-2) * (2 - 2**-M)
        (So the (largest / smallest) denormal number ratio is roughly 2**(2**E-1))

    Examples:
    * torch.half is M, E, A = 10, 5, -14; (modulo that the real format uses max exponent for NaN)
    * torch.float8_e5m2 is M, E, A = 2, 5, -14 (also modulo that the real format uses max exponent for NaN)
    * torch.float8_e4m3fn is M, E, A = 3, 4, -6 (modulo that the real format uses max denormal for NaN)
    * int8 is M, E, A = 7, 0, 7; this represents ±{0, 1, ..., 127}. (modulo that real int8 also has -128)
    * ternary weights are M, E, A = 0, 1, 0, this represents {-1, 0, +1}.
    * you could even have M, E, A = 0, 3, -2; this represents ±{0, 0.25, 0.5, 1, 2, 4, 8, 16}.

    In every case, the number of represented positive numbers is 2**(M+E).
    """

    xp = x.detach().abs()
    mantissa, exponent = torch.frexp(xp)
    mantissa *= 2 # bring mantissa into the range [1, 2) instead of [0.5, 1)
    exponent -= 1
    sign = mantissa.sign()
    mantissa = mantissa.abs()
    exponent = exponent.to(x.dtype)
    assert (2**exponent * mantissa == xp).all(), x[2**exponent * mantissa != xp]

    # Round mantissa to given precision
    mantissa = (1 + 2**-M * ((mantissa - 1) * 2**M).round())

    # Handle subnormals separately
    mask = (exponent < A)
    mantissa[mask] = (xp[mask] * 2**(M-A)).round() / 2**M
    exponent[mask] = A

    mask = (mantissa == 2)
    mantissa[mask] = 1
    exponent[mask] = exponent[mask] + 1

    # Truncate top of range
    if E > 0:
        B = A+2**E-2
        mask = (exponent > B)
        mantissa[mask] = 2 - 2**-M
        exponent[mask] = B
    else: # zero-bit exponent case: so we have only subnormal numbers (like an int8)
        mask = (mantissa >= 1)
        mantissa[mask] = 1 - 2**-M
        exponent[mask] = A

    y = x.sign() * 2**exponent * mantissa
    return y + (x - x.detach())

class CastedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        #w = self.weight.data
        #torch.nn.init.dirac_(w[:w.size(1)])
        
    def forward(self, x):
        if len(self.weight) == 24:
            return F.conv2d(x, self.weight, padding=self.padding, bias=self.bias)
        # Uses the casted weights for both forward and backward pass,
        # while the updates go to the high precision weights. This can be thought of
        # as a "straight-through estimator".

        s = self.weight[0].numel()**0.5
        #s = 1 / self.weight.data.abs().mean()

        M, E, A = hyp['net']['MEA']
        w = (1/s) * cast_tensor(s * self.weight, M, E, A)
        return F.conv2d(x, w, padding=self.padding, bias=self.bias)

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = CastedConv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = CastedConv(channels_out, channels_out)
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

class Conv(nn.Conv2d): # this is just for the first layer
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

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
    w = hyp['net']['width_factor']
    w1 = int(w*widths['block1'])
    w2 = int(w*widths['block2'])
    w3 = int(w*widths['block3'])
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, w1),
        ConvGroup(w1, w2),
        ConvGroup(w2, w3),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(w3, 10, bias=False),
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

    model = make_net()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    total_train_steps = epochs * len(train_loader)

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if len(p.shape) < 4 and p.requires_grad and 'norm' in n]
    other_params = [p for n, p in model.named_parameters() if len(p.shape) < 4 and p.requires_grad and 'norm' not in n]
    optimizer1 = RenormSGD(filter_params, lr=hyp['opt']['filter_lr'], momentum=hyp['opt']['momentum'], nesterov=True)
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
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
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad()
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
    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'], altflip=True)
    test_loader = CifarLoader('/tmp/cifar10', train=False)

    # (M, E, A)
    cast_settings = [
        # baselines
        (10, 5, -14),
        (2, 5, -14),
        # 1 bit (not counting the sign bit)
        (0, 1, 0), # {0, 1}
        # 2 bits
        (0, 2, -1), # {0, 1/2, 1, 2}
        (0, 2, 0), # {0, 1, 2, 4}
        #(1, 1, 0), # {0, 1/2, 1, 3/2}
        #(1, 1, 1), # {0, 1, 2, 3}
        # 3 bits
        #(2, 1, 0), # {0, 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4}
        (1, 2, -1), # {0, 1/4, 1/2, 3/4, 1, 3/2, 2, 3}
        (0, 3, -3), # {0, 1/8, 1/4, 1/2, 1, 2, 4, 8}
        # 4 bits
        #(2, 2, 0), # {0, 1/4, 1/2, 3/4, 1, ..., 4, 5, 6, 7}
        #(1, 3, -3), # {0, 1/16, 1/8, 3/16, ..., 4, 6, 8, 12}
        #(0, 4, -9), # {0, 2**-9, ..., 1/2, 1, 2, ..., 2**5}
    ]
    width_settings = [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
    ]
    for epochs in [20]:
        for w in width_settings:
            for mea in cast_settings:
                hyp['opt']['epochs'] = epochs
                hyp['net']['width_factor'] = w
                hyp['net']['MEA'] = mea
                res = torch.std_mean(torch.tensor([evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level']) for _ in range(100)]))
                obj = dict(w=w, mea=mea, res=res, epochs=epochs)
                import os
                os.makedirs('logs', exist_ok=True)
                import uuid
                torch.save(obj, 'logs/%s.pt' % uuid.uuid4())

