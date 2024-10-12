"""
clean_spectral.py

In n=50: (train on 40K eval 20K)
Mean: 0.9385    Std: 0.0009

Just change back to 50K/10K split and epochs=8 to recover clean variant of airbench94_muon.
"""
from math import ceil
from tqdm import tqdm
import torch
from torch import nn
from airbench import CifarLoader, evaluate
torch.backends.cudnn.benchmark = True

#############################################
#           Spectral SGD-momentum           #
#############################################

def spectral_norm(G, steps=25, eps=1e-7):
    if len(G.shape) == 4:
        G = G.reshape(len(G), -1)
    G_norm = G.norm()
    X = G.bfloat16() / G_norm
    if G.size(0) > G.size(1):
        X = X.T
    M = X @ X.T
    v = torch.randn(X.size(0), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        v = M @ (v / v.norm())
    return G_norm * v.norm()**0.5

#@torch.compile
def zeroth_power_via_newton(G, steps=15):

    orig_dtype = G.dtype
    G = G.bfloat16()

    d1, d2 = G.shape
    d = min(d1, d2)
    I = torch.eye(d, device=G.device, dtype=G.dtype)

    # store the smaller of the squares as S
    S = G @ G.T if d1 < d2 else G.T @ G
    S_norm = torch.linalg.matrix_norm(S, ord='fro') # there is freedom here. See Lakic (1998) Thm 2.3
    #S_norm = spectral_norm(S)
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

#@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=9, eps=1e-7):
    assert len(G.shape) == 2
    #a, b, c = (3.4445, -4.7750,  2.0315)
    a, b, c = (3.1866, -4.4189,  2.1207)
    #a, b, c = (2.613, -3.2274, 1.6137)
    #X = 0.9 * G.bfloat16() / (spectral_norm(G) + eps) # ensure top singular value <= 1
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

def zeroth_power_via_svd(G, steps=None):
    U, S, V = G.float().svd()
    return U @ S.diag() @ V.T

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
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
                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']

                ortho = zeroth_power_via_newtonschulz5
                #ortho = zeroth_power_via_newton
                #ortho = zeroth_power_via_svd

                ### Post-orthogonalize
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) # Nesterov momentum
                g = ortho(g.reshape(len(g), -1), steps=9).view(g.shape)

                #g /= spectral_norm(g)
                ## Normalize and upate the weight
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                p.data.add_(g, alpha=-lr)

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
    widths = dict(block1=64, block2=256, block3=256)
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
        Mul(1/9),
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

def main(run, model):

    epochs = 10
    lr = 1.655910/2000
    wd = 0.003821

    lr_biases = lr * 64
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=2000,
                               aug=dict(flip=True, translate=2, cutout=0), altflip=True)
    test_loader.images = torch.cat([test_loader.images, train_loader.images[40000:]])
    test_loader.labels = torch.cat([test_loader.labels, train_loader.labels[40000:]])
    train_loader.images = train_loader.images[:40000]
    test_loader.labels = test_loader.labels[:40000]

    total_train_steps = ceil(len(train_loader) * epochs)
    current_steps = 0

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()
    raw_model = (model._orig_mod if hasattr(model, '_orig_mod') else model)
    raw_model[0].weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    # Create optimizers for train whiten bias stage
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    whiten_bias = raw_model[0].bias
    fc_layer = raw_model[-2].weight
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=[fc_layer, whiten_bias], lr=lr, weight_decay=wd/lr)]
    optimizer1 = Muon(filter_params, lr=0.24, momentum=0.6)
    optimizer2 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
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

    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    print(tta_val_acc)
    return tta_val_acc


model = make_net()
#model = torch.compile(model, mode='max-autotune')
accs = torch.tensor([main(run, model) for run in tqdm(range(50))])
print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

