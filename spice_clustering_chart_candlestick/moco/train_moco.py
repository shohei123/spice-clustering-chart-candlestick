# CLI実行用
# import sys
# from rich import print
# sys.path.append("/home/shohei/crypto/my_code/custom/spice_clustering_chart_candlestick/spice_clustering_chart_candlestick")

# PyTorch Lightning
import pytorch_lightning as pl

# MoCo
import moco.callbacks as callbacks
import moco.models as moco_models
from moco.build_model import build_model
from moco.data_loader import ChartDataLoader

# general
from argparse import ArgumentParser
import pretty_errors


def main(args):

    if args.seed is not None:
        pl.seed_everything(seed=args.seed, workers=True)

    cdl = ChartDataLoader(
        crop_min=args.crop_min,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        persistent=args.persistent,
        workers=args.workers,
    )

    num_batches = len(cdl)

    model = build_model(
        arch=args.arch,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        moco_dim=args.moco_dim,
        moco_mlp_dim=args.moco_mlp_dim,
        moco_momentum=args.moco_momentum,
        moco_momentum_cosine=args.moco_momentum_cosine,
        moco_temperature=args.moco_temperature,
        momentum=args.momentum,
        num_batches=num_batches,
        optimizer_type=args.optimizer_type,
        stop_grad_conv1=args.stop_grad_conv1,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer.from_argparse_args(
        accelerator=args.accelerator,
        auto_scale_batch_size=args.auto_scale_batch_size,
        callbacks=[callbacks.PrintCallback()],
        deterministic=args.deterministic,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        strategy=args.strategy,
        sync_batchnorm=args.sync_batchnorm,
    )

    trainer.fit(
        model=model,
        train_dataloaders=cdl,
    )


if __name__ == "__main__":

    parser = ArgumentParser()

    # 学習環境の引数
    parser.add_argument(
        "--batch_size",
        default=256,  # default 4096
        type=int,
        help="",
    )

    parser.add_argument(
        "--data_dir",
        default="../datasets/chart",
        type=str,
        help="MoCoに学習させるチャート画像を格納したディレクトリパス",
    )

    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.6,
        type=float,
        help="",
    )

    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="LARSオプティマイザーで使用するモメンタム",
    )

    parser.add_argument(
        "--persistent",
        action="store_true",
        help="ddp_spawnかつworker>0の場合は、ボトルネック解消のために有効にする"
    ),

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="",
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-6,
        type=float,
        help="LARSオプティマイザーで使用する重み減衰値",
    )

    parser.add_argument(
        "--workers",
        default=32,
        type=int,
        help="全GPUの合計ワーカー数",
    )

    parser.add_argument(
        "--worm_up_epochs",
        default=10,
        type=int,
        help="学習率の更新で使うウォームアップ値",
    )

    # Vision Transformerの設定
    parser.add_argument(
        "--stop_grad_conv1",
        action="store_true",
        help="stop-grad after first conv, or patch embedding"
    ),

    # 画像の加工
    parser.add_argument(
        "--crop_min",
        default=0.08,
        type=float,
        help="画像の拡縮処理の最小値",
    )

    # モデルの引数
    parser = moco_models.MoCo.add_model_specific_args(parser)

    # トレーナーの引数
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
