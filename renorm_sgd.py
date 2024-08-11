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

        #shape = [len(d_p)]+[1]*(len(d_p.shape)-1)
        #d_p_hat = d_p / d_p.reshape(len(d_p), -1).norm(dim=1).view(*shape)
        #param.div_(param.data.reshape(len(param), -1).norm(dim=1).view(*shape))
        param.data.div_(param.data.norm())
        param.data.add_(d_p, alpha=-lr / d_p.norm())

