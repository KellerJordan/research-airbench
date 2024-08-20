"""
casted_airbench.py

Accuracy: 93.95 in n=50

At various precisions:
* bits=3 -> 93.95 (n=50)
* bits=2 -> 93.95 (n=100)
* bits=1 -> 93.86 (n=50)
* bits=0 -> 93.67 (n=50)

What if we leave the whitening convolution in bits=2?
* a=2**-8 bits=0 -> 93.76 (n=30)
* bits=0 -> 93.73 (n=50)
What if we leave all the convolutions with <= 64 filters in bits=2?
* a=2**-8 bits=0 -> 93.80 (n=20)

Now always with bits=2, ablating over the bounds
* (lg a, lg b) = (-inf, 2) -> 93.90 (n=50)
* (lg a, lg b) = (-inf, 3) -> 93.93 (n=50)
* (lg a, lg b) = (-8, 3) -> 93.91 (n=50)
* (lg a, lg b) = (-7, 3) -> 93.78 (n=50)
* (lg a, lg b) = (-6, 3) -> 93.29 (n=50)

What if we do stochastic rounding of the weights every batch, instead of deterministic?
* It's terrible. Not sure if it would be better if we did it at the example level.

What if we leave the weights 2-bit, and also cast the inputs?
* bits=5 -> 93.89 (n=50)
* bits=4 -> 93.89 (n=50)
* bits=3 -> 93.82 (n=50)
* bits=2 -> 93.63 (n=50)

What if we don't upper-bound the first weight (which is the whitening layer)?
* (lg a, lg b) = (-inf, 1) -> 93.90 (n=20)
* (lg a, lg b) = (-inf, 0) -> 93.98 (n=20)
* (lg a, lg b) = (-inf, -1) -> 93.95 (n=50)
* (lg a, lg b) = (-inf, -2) -> 93.73 (n=10)
* (lg a, lg b) = (-inf, -3) -> 93.58 (n=10)
* (lg a, lg b) = (-10, 0) -> 93.94 (n=50)

Final settings. Assume w_bits=2 a=2**-10 b=2**0 unless otherwise specified.
* [default -> 93.94 (n=50)]
* x_bits=5 a=2**-8 -> 93.84 (n=50)
* w_bits=1 x_bits=5 -> 93.83 (n=50)
* a=2**-8 -> 93.90 (n=50)
* x_bits=8 -> 93.95 (n=50)
* x_bits=5 -> 93.89 (n=50)
* x_bits=4 -> 93.89 (n=50)
* x_bits=3 -> 93.80 (n=50)

Now let's remove the dirac initialization so that we can shrink the range.
Parametrize a and b so that they are multiplied by 1/d**0.5 where d is the number
of incoming channels (so the average PT init would be (1/(3*3)*3)**0.5 times this).
Again let w_bits=2 and unlimited range be the default.
* default -> 93.56 (n=50)
* a=2**-4 b=2**4 -> 93.49 (n=50)
* a=2**-10 b=2**10 -> 93.55 (n=50)
* a=2**-6 b=2**2 -> 93.55 (n=25)
* a=2**-6 b=2**0 -> 93.61 (n=25)
* a=2**-6 b=2**-1 -> 93.52 (n=50)
* a=2**-5 b=2**-1 -> 93.51 (n=50)
* a=2**-6 b=2**-2 -> 93.35 (n=25)
* a=2**-3 b=2**-1 -> 93.29 (n=25)

Therefore overall we can reach the following conclusions.
* We need 2 bits of precision for the weights and 4 bits for the activations.
    - Losing 1 bit on either -> -0.1% accuracy.
* We need 2 bits of range for the weights.
    - Reducing to 1 bit of range -> -0.25%

Now with new tech for the bottom of the range. If you're below a then round to either 0 or a.
Previously it was just rounding down to 0 always.
* default -> 93.94 (n=50)
* a=2**-6 -> 93.56 (n=25) [this improves significantly over the prior result of 93.29]

Now also with removing dirac and relative-to-sqrt tech etc.
* a=2**-5 b=2**-1 -> 93.55 (n=25)
* a=2**-3 b=2**-1 -> 93.40 (n=25) [again improving over prior result]
oops the below are mistaken because I didn't keep the whitening layer in bits=2
* bits=1 a=2**-5 b=2**-1 -> 93.49 (n=25)
* bits=1 a=2**-3 b=2**-1 -> 93.36 (n=25)
* bits=0 a=2**-5 b=2**-1 -> 93.26 (n=25)
* bits=0 a=2**-3 b=2**-1 -> 93.22 (n=25)
* bits=0 a=2**-2 b=2**-1 -> 92.81 (n=25)
* bits=0 a=2**-3 b=2**-2 -> 93.06 (n=25)
now with whitening layer in bits=2
* bits=0 a=b=2**-2 -> 92.66 (10) [wd=0.01 -> 92.75 (25)] [~triple compute, adjust lr -> 93.75 (25)] [2.25x compute, adjust lr -> 93.59]
* bits=0 a=b=2**-3 -> 92.67 (10) [wd=0.01 -> 92.69 (25)] [triple compute, adjust lr -> 93.79 (25)]

These last networks were ternary. They apparently perform as well as ~2.25x-larger full precision networks.
This the same result that we can see in the training loss reported by the 1.58-bit paper.
https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
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
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'scaling_factor': 1/9,
        'tta_level': 2,
        'conv_precision': {
            'bits': 2, # bits per binary level: bits=3 means we can represent 8 values in the range [1, 2), 8 values in [2, 4) etc
            'upper_bound': 2**0,
            'lower_bound': 2**-10,
        }
    },
}

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

# Reduces the precision of x to the given number of bits.
# The number of bits for the exponent is unchanged.
# Note that this is a "simulation" of actual low-precision casting, i.e., we leave the weights in their
# incoming hardware datatype; what changes is just that we round the values to the nearest valid
# values at the lower precision.
def cast_tensor(tensor, bits, a, b, eps=1.7881e-07):
    x = tensor.clone()
    if b is not None:
        x = x.sign() * x.abs().clamp(0, b)
    x[x == 0] = eps

    log2 = torch.tensor(2.).log().cuda()
    exp = (x.float().abs().log() / log2).floor()
    frac = x * 2**-exp
    frac_newfp = frac.sign() * (1 + torch.round(2**bits * (frac.abs() - 1)) / 2**bits)

    casted_x = (frac_newfp * 2**exp).to(x.dtype)
    if a is not None:
        casted_x[x.abs() < a/2] = 0
        mask = (x.abs() >= a/2) & (x.abs() < a)
        casted_x[mask] = a * x.sign()[mask]

    # let the gradients flow through the original tensor
    return casted_x.detach() + (tensor - tensor.detach())

class CastedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
        
    def forward(self, x):
        bits = hyp['net']['conv_precision']['bits']
        a = hyp['net']['conv_precision']['lower_bound']
        b = hyp['net']['conv_precision']['upper_bound']
        if len(self.weight) == 24:
            b = None
        if bits is not None:
            # This uses the casted weights for both forward and backward pass,
            # while the updates go to the actual high precision weights
            w = cast_tensor(self.weight, bits, a, b)
            return F.conv2d(x, w, padding=self.padding, bias=self.bias)
        else:
            return F.conv2d(x, self.weight, padding=self.padding, bias=self.bias)

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
        CastedConv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
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

    model = make_net()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    total_train_steps = epochs * len(train_loader)

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.2)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return (1 - frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    current_steps = 0
    train_loader.epoch = 0
    model.train()
    from tqdm import tqdm
    for epoch in tqdm(range(epochs)):
        for inputs, labels in train_loader:

            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1
            if current_steps == total_train_steps:
                break

    return model

if __name__ == '__main__':
    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'], altflip=True)
    test_loader = CifarLoader('/tmp/cifar10', train=False)

    print(evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level']))
    print(torch.std_mean(torch.tensor([evaluate(train(train_loader), test_loader, tta_level=hyp['net']['tta_level'])
                                       for _ in range(50)])))

