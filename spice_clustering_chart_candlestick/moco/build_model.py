# PyTorch
import torchvision.models as torchvision_models

# Vision Transformer
import moco.vits as vits

# MoCo
import moco.models as moco_models

# general
from functools import partial
from omegaconf import DictConfig


def build_model(
    cfg: DictConfig,
    cfg_hparams,
):

    print("モデルの作成 '{}'".format(cfg.model.arch))

    # archとしてVision Transformerが指定されている場合
    # 返り値のmodelは、クラスから生成されたインスタンス
    if cfg.model.arch.startswith("vit"):
        model = moco_models.MoCo_ViT(
            base_encoder=partial(
                vits.__dict__[cfg.model.arch],
                stop_grad_conv1=cfg.model.stop_grad_conv1
            ),
            cfg=cfg,
            cfg_hparams=cfg_hparams,
        )

    # torchvisionの組み込みモデルが指定されている場合
    else:
        model = moco_models.MoCo_ResNet(
            base_encoder=partial(
                torchvision_models.__dict__[cfg.model.arch],
                zero_init_residual=True
            ),
            cfg=cfg,
            cfg_hparams=cfg_hparams,
        )

    return model
