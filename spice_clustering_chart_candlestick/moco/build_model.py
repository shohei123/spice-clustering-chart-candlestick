# PyTorch
import torchvision.models as torchvision_models

# Vision Transformer
import moco.vits as vits

# MoCo
import moco.models as moco_models

# general
from functools import partial


def build_model(
    epochs,
    learning_rate,
    moco_dim,
    moco_mlp_dim,
    moco_momentum,
    moco_momentum_cosine,
    moco_temperature,
    momentum,
    num_batches,
    optimizer_type,
    weight_decay,
    arch: str = "vit_base",
    stop_grad_conv1: bool = True,
):

    print("モデルの作成 '{}'".format(arch))

    # archとしてVision Transformerが指定されている場合
    # 返り値のmodelは、クラスから生成されたインスタンス
    if arch.startswith("vit"):
        model = moco_models.MoCo_ViT(
            base_encoder=partial(
                vits.__dict__[arch],
                stop_grad_conv1=stop_grad_conv1
            ),
            epochs=epochs,
            learning_rate=learning_rate,
            moco_dim=moco_dim,
            moco_mlp_dim=moco_mlp_dim,
            moco_momentum=moco_momentum,
            moco_momentum_cosine=moco_momentum_cosine,
            moco_temperature=moco_temperature,
            momentum=momentum,
            num_batches=num_batches,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
        )

    # torchvisionの組み込みモデルが指定されている場合
    else:
        model = moco_models.MoCo_ResNet(
            base_encoder=partial(
                torchvision_models.__dict__[arch],
                zero_init_residual=True
            ),
            epochs=epochs,
            learning_rate=learning_rate,
            moco_dim=moco_dim,
            moco_mlp_dim=moco_mlp_dim,
            moco_momentum=moco_momentum,
            moco_momentum_cosine=moco_momentum_cosine,
            moco_temperature=moco_temperature,
            momentum=momentum,
            num_batches=num_batches,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
        )

    return model
