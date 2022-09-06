import math


def adjust_learning_rate(optimizer, epoch, args):
    """ウォームアップ後、半周期のコサインで学習速度を減衰させる"""

    # ウォームアップ中は、意図的に学習率を抑えておく
    # 大きなバッチサイズで学習しても、精度が落ちにくくなる
    if epoch < args.warm_up_epochs:
        lr = args.learning_rate * epoch / args.warm_up_epochs

    # コサイン学習率スケジュールのウォームアップ考慮版
    # 学習率は、滑り台のように緩やかに低下していく
    else:
        lr = args.learning_rate * 0.5 * (
            1. + math.cos(
                math.pi * (epoch - args.warm_up_epochs) /
                (args.epochs - args.warm_up_epochs)
            )
        )

    # 計算グラフの変数ごとに固有のパラメーター群を持つので、
    #  forループで1つ1つの変数の「param_group」を取得する
    for param_group in optimizer.param_groups:
        param_group["learning_rate"] = lr


def adjust_moco_momentum(epoch, args):
    """現在のエポックに基づきmomentumを調整"""

    # コサインスケジュールを取り入れて、緩やかにmomentumを低下させる
    momentum = 1. - 0.5 * \
        (1. + math.cos(math.pi * epoch / args.epochs)) * (
            1. - args.moco_momentum
        )
    return momentum
