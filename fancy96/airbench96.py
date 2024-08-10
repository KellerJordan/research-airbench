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

from model import make_net, reinit_net, init_whitening_conv
from utils import evaluate, set_random_state, InfiniteCifarLoader

torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'train_epochs': 50.0,
        'batch_size': 1024,
        'lr': 9.0,               # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.012,   # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,     # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3, # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 12,
    },
    'proxy': {
        'widths': {
            'block1': 64,
            'block2': 64,
            'block3': 64,
        },
        'depth': 2,
        'scaling_factor': 1/9,
    },
    'net': {
        'widths': {
            'block1': 128,
            'block2': 384,
            'block3': 512,
        },
        'depth': 3,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#                Training                  #
############################################

def train_proxy(hyp, model):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = InfiniteCifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=0, order_seed=0)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model)
    print('Proxy parameters:', sum(p.numel() for p in model.parameters()))
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Initialize the whitening layer using training images
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model._orig_mod[0], train_images)

    masks = []

    for indices, inputs, labels in train_loader:

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            model.train()

        outputs = model(inputs)
        loss1 = loss_fn(outputs, labels)
        mask = torch.zeros(len(inputs)).cuda().bool()
        mask[loss1.argsort()[-512:]] = True
        masks.append(mask)
        loss = (loss1 * mask.float()).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_steps += 1
        if current_steps == total_train_steps:
            break

    return masks

def main(run, hyp, model_proxy, model_trainbias, model_freezebias):

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
    test_loader = InfiniteCifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=0, order_seed=0)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model_trainbias)
    print('Main model parameters:', sum(p.numel() for p in model_trainbias.parameters()))
    current_steps = 0

    norm_biases = [p for k, p in model_trainbias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_trainbias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_trainbias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    norm_biases = [p for k, p in model_freezebias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_freezebias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_freezebias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, get_lr)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer_freezebias, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model_trainbias)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Do a small proxy run to collect masks for use in fullsize run
    print('Training small proxy...')
    starter.record()
    masks = iter(train_proxy(hyp, model_proxy))
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for indices, inputs, labels in train_loader:

        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if current_steps == 0:
            model = model_trainbias
            optimizer = optimizer_trainbias
            scheduler = scheduler_trainbias
        elif epoch == hyp['opt']['whiten_bias_epochs'] * steps_per_epoch:
            model = model_freezebias
            optimizer = optimizer_freezebias
            scheduler = scheduler_freezebias
            model.load_state_dict(model_trainbias.state_dict())
            optimizer.load_state_dict(optimizer_trainbias.state_dict())
            scheduler.load_state_dict(scheduler_trainbias.state_dict())

        ####################
        #     Training     #
        ####################

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            starter.record()
            model.train()

        mask = next(masks)
        inputs = inputs[mask]
        labels = labels[mask]
        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_steps += 1
        if current_steps % 5 == 0:
            lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        if current_steps == total_train_steps:
            if lookahead_state is not None:
                lookahead_state.update(model, decay=1.0)

        if (current_steps % steps_per_epoch == 0) or (current_steps == total_train_steps):
            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)

            ####################
            #    Evaluation    #
            ####################

            # Save the accuracy and loss from the last training batch of the epoch
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)
            print_training_details(locals(), is_final_entry=False)
            run = None # Only print the run number once

        if current_steps == total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    model_proxy = make_net(hyp['proxy'])
    #model_proxy[0].bias.requires_grad = False
    model_trainbias = make_net(hyp['net'])
    model_freezebias = make_net(hyp['net'])
    model_freezebias[0].bias.requires_grad = False
    model_proxy = torch.compile(model_proxy, mode='max-autotune')
    model_trainbias = torch.compile(model_trainbias, mode='max-autotune')
    model_freezebias = torch.compile(model_freezebias, mode='max-autotune')

    print_columns(logging_columns_list, is_head=True)
    accs = torch.tensor([main(run, hyp, model_proxy, model_trainbias, model_freezebias)
                         for run in range(15)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))

