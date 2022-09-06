import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(
        self,
        params,
        lr,
        weight_decay,
        momentum,
        trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for pram_group in self.param_groups:
            for param in pram_group["params"]:
                dp = param.grad

                if dp is None:
                    continue

                if param.ndim > 1:  # if not normalization gamma/beta or bias
                    dp = dp.add(param, alpha=pram_group["weight_decay"])
                    param_norm = torch.norm(param)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.,
                        torch.where(
                            update_norm > 0,
                            (
                                pram_group["trust_coefficient"] *
                                param_norm / update_norm
                            ),
                            one
                        ),
                        one
                    )
                    dp = dp.mul(q)

                param_state = self.state[param]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(param)
                mu = param_state["mu"]
                mu.mul_(pram_group["momentum"]).add_(dp)
                param.add_(mu, alpha=-pram_group["lr"])
