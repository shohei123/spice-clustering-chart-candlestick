import math


def adjust_moco_momentum(
    max_epochs,
    moco_momentum,
    modified_epoch,
):
    """現在のエポックに基づきmomentumを調整"""

    # コサインスケジュールを取り入れて、緩やかにmomentumを低下させる
    momentum = 1. - 0.5 * (1. + math.cos(math.pi * modified_epoch / max_epochs)) * (1. - moco_momentum)
    return momentum
