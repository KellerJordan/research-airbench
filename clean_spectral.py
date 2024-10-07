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

torch.backends.cudnn.benchmark = True

from airbench import CifarLoader, evaluate

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
}

#############################################
#           Spectral SGD-momentum           #
#############################################

@torch.compile
def zeroth_power_via_newton(G, steps=9):

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
        if step < steps-1: # optimization suggested by @EitanTurok https://x.com/EitanTurok/status/1839754807696855333
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
    def __init__(self, num_features, momentum, eps=1e-12,
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
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
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

def make_net():
    widths = dict(block1=64, block2=256, block3=256)
    batchnorm_momentum = 0.6
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(1/9),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Training                  #
############################################

def main(run, model_trainbias, model_freezebias):

    batch_size = hyp['opt']['batch_size']
    epochs = 8
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

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=2000, aug=dict(flip=True, translate=2))
    total_train_steps = ceil(len(train_loader) * epochs)

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    reinit_net(model_trainbias)
    current_steps = 0

    # Create optimizers for train whiten bias stage
    model = model_trainbias
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    whiten_bias = model._orig_mod[0].bias
    fc_layer = model._orig_mod[-2].weight
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=[fc_layer], lr=lr, weight_decay=wd/lr)]
    optimizer1 = SpectralSGDM(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    #optimizer1 = torch.optim.SGD(filter_params, lr=lr, weight_decay=wd/lr, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer3 = torch.optim.SGD([whiten_bias], lr=lr, weight_decay=wd/lr, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer1_trainbias = optimizer1
    optimizer2_trainbias = optimizer2
    optimizer3_trainbias = optimizer3
    # Create optimizers for frozen whiten bias stage
    model = model_freezebias
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    fc_layer = model._orig_mod[-2].weight
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=[fc_layer], lr=lr, weight_decay=wd/lr)]
    optimizer1 = SpectralSGDM(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    #optimizer1 = torch.optim.SGD(filter_params, lr=lr, weight_decay=wd/lr, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer1_freezebias = optimizer1
    optimizer2_freezebias = optimizer2
    # Make learning rate schedulers for all 5 optimizers
    def get_lr(step):
        return 1 - step / total_train_steps
    scheduler1_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer1_trainbias, get_lr)
    scheduler2_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer2_trainbias, get_lr)
    scheduler3_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer3_trainbias, get_lr)
    scheduler1_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer1_freezebias, get_lr)
    scheduler2_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer2_freezebias, get_lr)

    # Initialize the whitening layer using training images
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)

    for epoch in range(ceil(epochs)):

        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if epoch == 0:
            model = model_trainbias
            optimizers = [optimizer1_trainbias, optimizer2_trainbias, optimizer3_trainbias]
            schedulers = [scheduler1_trainbias, scheduler2_trainbias, scheduler3_trainbias]
        elif epoch == hyp['opt']['whiten_bias_epochs']:
            model = model_freezebias
            old_optimizers = optimizers
            old_schedulers = schedulers
            optimizers = [optimizer1_freezebias, optimizer2_freezebias]
            schedulers = [scheduler1_freezebias, scheduler2_freezebias]
            model.load_state_dict(model_trainbias.state_dict())
            for i, (opt, sched) in enumerate(zip(optimizers, schedulers)):
                opt.load_state_dict(old_optimizers[i].state_dict())
                sched.load_state_dict(old_schedulers[i].state_dict())

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            current_steps += 1
            if current_steps >= total_train_steps:
                break

    return evaluate(model, test_loader, tta_level=2)

if __name__ == "__main__":

    model_trainbias = make_net()
    model_freezebias = make_net()
    model_freezebias[0].bias.requires_grad = False
    model_trainbias = torch.compile(model_trainbias, mode='max-autotune')
    model_freezebias = torch.compile(model_freezebias, mode='max-autotune')

    from tqdm import tqdm
    accs = torch.tensor([main(run, model_trainbias, model_freezebias) for run in tqdm(range(50))])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

