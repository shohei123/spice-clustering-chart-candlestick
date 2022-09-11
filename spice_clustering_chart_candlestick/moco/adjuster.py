import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class CosineAnnealingWarmupRestarts(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float,
        gamma: float,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # 最初の周期までのステップ数
        self.cycle_mult = cycle_mult  # 次の周期に達するまでのステップ数の増大倍率
        self.base_max_lr = max_lr  # 最初の周期の学習率の上限
        self.max_lr = max_lr  # 現在の周期における学習率の暫定な上限
        self.min_lr = min_lr  # 現在の周期における学習率の暫定な下限
        self.warmup_steps = warmup_steps  # ウォームアップ期間のステップ数
        self.gamma = gamma  # 1周期ごとの学習率の減衰倍率

        self.cur_cycle_steps = first_cycle_steps  # 現在の周期における必要ステップ数
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # 現在の周期におけるステップ数

        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, last_epoch, verbose
        )

        # オプティマイザー内のパラメーターの学習率を最小学習率で初期化
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning
            )
        if self.step_in_cycle == -1:
            return self.base_lrs

        # ウォームアップ数に近づくほど、だんだんと上限の学習率に近い値を返却
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]

        # ウォームアップ以降は、ウォームアップ分だけ差し引いたステップ数でコサインアニーリングを適用
        else:
            return [
                base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

            # 次の周期に入る場合
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                # 現在の周期におけるステップ数を初期化
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps  # 次の周期に要するステップ数を更新

        # チェックポイントから復元するなど、epochに引数が与えられる場合
        else:
            # エポック数が第1周期のステップ数を超えている場合
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)

        # PyTorchのlr_schedulerに見られるので、ひとまず導入
        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def adjust_moco_momentum(
    max_epochs,
    moco_momentum,
    modified_epoch,
):
    """現在のエポックに基づきmomentumを調整"""

    # コサインスケジュールを取り入れて、緩やかにmomentumを低下させる
    momentum = 1. - 0.5 * (1. + math.cos(math.pi * modified_epoch / max_epochs)) * (1. - moco_momentum)
    return momentum
