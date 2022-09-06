# PyTorch
import torchvision.models as torchvision_models

# Vision Transformer
import vits

# MoCo
import moco.models as moco_models

# general
from functools import partial


def build_model(args):

    print("モデルの作成 '{}'".format(args.arch))

    # archとしてVision Transformerが指定されている場合
    # 返り値のmodelは、クラスから生成されたインスタンス
    if args.arch.startswith("vit"):
        model = moco_models.MoCo_ViT(
            partial(
                vits.__dict__[args.arch],
                stop_grad_conv1=args.stop_grad_conv1
            ),
            args,
        )

    # torchvisionの組み込みモデルが指定されている場合
    else:
        model = moco_models.MoCo_ResNet(
            partial(
                torchvision_models.__dict__[args.arch],
                zero_init_residual=True
            ),
            args,
        )

    print("モデルの詳細'{}'".format(model))

    return model
