import math


def adjust_learning_rate(
    epochs,
    learning_rate,
    modified_epoch,
    optimizer,
    warm_up_epochs,
):
    """ウォームアップ後、半周期のコサインで学習速度を減衰させる"""

    # ウォームアップ中は、意図的に学習率を抑えておく
    # 大きなバッチサイズで学習しても、精度が落ちにくくなる
    if modified_epoch < warm_up_epochs:
        lr = learning_rate * modified_epoch / warm_up_epochs

    # コサイン学習率スケジュールのウォームアップ考慮版
    # 学習率は、滑り台のように緩やかに低下していく
    else:
        lr = learning_rate * 0.5 * (
            1. + math.cos(
                math.pi * (modified_epoch - warm_up_epochs) /
                (epochs - warm_up_epochs)
            )
        )

    # 計算グラフの変数ごとに固有のパラメーター群を持つので、
    #  forループで1つ1つの変数の「param_group」を取得する
    for param_group in optimizer.param_groups:
        param_group["learning_rate"] = lr


def adjust_moco_momentum(
    epochs,
    moco_momentum,
    modified_epoch,
):
    """現在のエポックに基づきmomentumを調整"""

    # コサインスケジュールを取り入れて、緩やかにmomentumを低下させる
    momentum = 1. - 0.5 * \
        (1. + math.cos(math.pi * modified_epoch / epochs)) * (
            1. - moco_momentum
        )
    return momentum
