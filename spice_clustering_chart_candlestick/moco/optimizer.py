import torch
from torch.optim.optimizer import Optimizer


# Pytorch Lightning BoltsのLARSアルゴリズムを移植
class LARS(Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>

    Args:
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
    eps (float, optional): eps for division denominator (default: 1e-8)
    """

    def __init__(
        self,
        params,
        eps: float,
        dampening: int,
        lr: float,
        momentum: float,
        nesterov: bool,
        trust_coefficient: float,
        weight_decay: float,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            eps=eps,
            dampening=dampening,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            weight_decay=weight_decay,
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional):
            A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
