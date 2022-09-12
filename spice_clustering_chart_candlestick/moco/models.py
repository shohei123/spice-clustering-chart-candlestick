# PyTorch
import torch

# Pytorch Lightning
import pytorch_lightning as pl

# MoCo
import moco.optimizer
import moco.adjuster as moco_adjuster
from moco.calc_contrastive_loss import contrastive_loss
from moco.build_mlp import build_mlp
from moco.lr_scheduler import CosineAnnealingWarmupRestarts

# general
from omegaconf import DictConfig


class MoCo(pl.LightningModule):
    def __init__(
        self,
        base_encoder,
        cfg: DictConfig,
        cfg_hparams,
    ):
        super(MoCo, self).__init__()
        self.save_hyperparameters("cfg_hparams")

        self.cfg = cfg

        # 引数のbase_encoderは、既製品のエンコーダーモデルを流用
        self.base_encoder = base_encoder(
            num_classes=self.cfg.model.moco_mlp_dim
        )
        self.momentum_encoder = base_encoder(
            num_classes=self.cfg.model.moco_mlp_dim
        )

        # 既製品のエンコーダーの最終層と置換する投影・推論層
        self._build_projector_and_predictor_mlps(
            self.cfg.model.moco_dim,
            self.cfg.model.moco_mlp_dim
        )

        # ベースエンコーダーのパラメーターをモーメンタムエンコーダーに移植
        for param_b, param_m in zip(
            self.base_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    # サブクラスで上書きするための空メソッド
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    # モーメンタムエンコーダーのパラメーター更新
    @torch.no_grad()
    def _update_momentum_encoder(self, momentum):
        for param_b, param_m in zip(
            self.base_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_m.data = param_m.data * momentum + param_b.data * (1. - momentum)

    # MoCo V3 の場合は、クエリとキューに2画像の変換を用いる
    # 損失の計算時には、異なる画像のクエリ・キューをペアにする
    def training_step(self, batch, batch_idx):

        # batchの中身は「2舞の画像を要素に持つリスト」と「ディレクトリ名の正解ラベル」なので、分離させる
        images, _ = batch

        # imagesの中にある2枚の画像を別々の変数に格納
        x1 = images[0]
        x2 = images[1]

        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(self.cfg.model.moco_momentum)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        loss = 0
        loss += contrastive_loss(q1, k2, self.cfg.model.moco_temperature)
        loss += contrastive_loss(q2, k1, self.cfg.model.moco_temperature)
        self.log("対照損失", loss, logger=True, prog_bar=True)

        if self.cfg.model.moco_momentum_cosine:
            self.moco_momentum = moco_adjuster.adjust_moco_momentum(
                max_epochs=self.cfg.trainer.max_epochs,
                moco_momentum=self.cfg.model.moco_momentum,
                modified_epoch=self.current_epoch + batch_idx / self.trainer.num_training_batches,
            )

        return loss

    def configure_optimizers(self):
        if self.cfg.optimizer.optimizer_type == "lars":
            optimizer = moco.optimizer.LARS(
                params=self.parameters(),
                eps=self.cfg.optimizer.eps,
                dampening=self.cfg.optimizer.dampening,
                lr=self.cfg.optimizer.learning_rate,
                nesterov=self.cfg.optimizer.nesterov,
                momentum=self.cfg.optimizer.momentum,
                trust_coefficient=self.cfg.optimizer.trust_coefficient,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.learning_rate,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        else:
            raise NotImplementedError("指定されているoptimizerを利用してください。")

        # ラーニングスケジュールの使い方について精査する
        scheduler = {
            "scheduler": CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=self.cfg.lr_scheduler.first_cycle_steps,
                cycle_mult=self.cfg.lr_scheduler.cycle_mult,
                gamma=self.cfg.lr_scheduler.gamma,
                max_lr=self.cfg.lr_scheduler.max_lr,
                min_lr=self.cfg.lr_scheduler.min_lr,
                warmup_steps=self.cfg.lr_scheduler.warmup_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, moco_dim, moco_mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]

        del self.base_encoder.fc, self.momentum_encoder.fc

        # projectors
        self.base_encoder.fc = build_mlp(
            2,
            hidden_dim,
            moco_mlp_dim,
            moco_dim
        )

        self.momentum_encoder.fc = build_mlp(
            2,
            hidden_dim,
            moco_mlp_dim,
            moco_dim
        )

        # predictor
        self.predictor = build_mlp(
            2,
            moco_dim,
            moco_mlp_dim,
            moco_dim,
            False
        )


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, moco_dim, moco_mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]

        # remove original fc layer
        del self.base_encoder.head, self.momentum_encoder.head

        # projectors
        self.base_encoder.head = build_mlp(
            3,
            hidden_dim,
            moco_mlp_dim,
            moco_dim
        )

        self.momentum_encoder.head = build_mlp(
            3,
            hidden_dim,
            moco_mlp_dim,
            moco_dim
        )

        # predictor
        self.predictor = build_mlp(
            2,
            moco_dim,
            moco_mlp_dim,
            moco_dim
        )
