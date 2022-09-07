# PyTorch
import torch

# Pytorch Lightning
import pytorch_lightning as pl

# MoCo
import moco.optimizer
from moco.calc_contrastive_loss import contrastive_loss
from moco.build_mlp import build_mlp
import moco.adjuster as moco_adjuster


class MoCo(pl.LightningModule):
    def __init__(
        self,
        base_encoder,
        args,
    ):
        super(MoCo, self).__init__()
        self.automatic_optimization = False
        self.args = args

        # 引数のbase_encoderは、既製品のエンコーダーモデルを流用
        self.base_encoder = base_encoder(
            num_classes=args.moco_mlp_dim
        )
        self.momentum_encoder = base_encoder(
            num_classes=args.moco_mlp_dim
        )

        # 既製品のエンコーダーの最終層と置換する投影・推論層
        self._build_projector_and_predictor_mlps(
            args.moco_dim,
            args.moco_mlp_dim
        )

        # ベースエンコーダーのパラメーターをモーメンタムエンコーダーに移植
        for param_b, param_m in zip(
            self.base_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    # モデル特有の引数の取得
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MoCoModel")

        parser.add_argument(
            "--arch",
            default="vit_base",
            type=str,
            help='エンコーダーで利用する既存モデル'
            'vit_small,vit_base,vit_conv_small, vit_conv_base'
            'torchvision_models'
        )

        parser.add_argument(
            "--moco_dim",
            type=int,
            default=256,
            help="MoCoによって得られる画像の分散表現の次元数"
        )

        parser.add_argument(
            "--moco_mlp_dim",
            type=int,
            default=4096,
            help="MoCoのMLP部分における隠れ層の次元数"
        )

        parser.add_argument(
            "--moco_momentum",
            type=float,
            default=0.99,
            help="MoCoエンコーダーのパラメーター更新で利用するモメンタム"
        )

        parser.add_argument(
            "--moco_temperature",
            type=float,
            default=1.0,
            help="MoCoエンコーダーのハイパーパラメーター"
        )

        parser.add_argument(
            "--moco_momentum_cosine",
            action="store_true",
            help="MoCoモメンタムを半周期コサインで漸増させるか"
        ),

        parser.add_argument(
            "--optimizer_type",
            default="lars",
            type=str,
            choices=["lars", "adamw"],
            help="オプティマイザーの種類「lars or adamw」"
        )

        return parent_parser

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
            param_m.data = param_m.data * momentum + \
                param_b.data * (1. - momentum)

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
            self._update_momentum_encoder(self.moco_momentum)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        loss = contrastive_loss(q1, k2) + contrastive_loss(q2, k1)
        self.log("対照損失", loss, logger=True, prog_bar=True)

        # オプティマイザーによる更新手続きについて、要確認
        opt = self.optimizers()
        opt.zero_grad()

        # オプティマイザー内の学習率を更新
        moco_adjuster.adjust_learning_rate(
            self.optimizer,
            self.current_epoch + batch_idx / self.args.num_batches,
            self.args
        )

        if self.args.moco_momentum_cosine:
            self.moco_momentum = moco.adjuster.adjust_moco_momentum(
                self.current_epoch + batch_idx / self.args.num_batches,
                self.args
            )

        self.manual_backward(loss)
        opt.step()

    def configure_optimizers(self):
        if self.args.optimizer_type == "lars":
            return moco.optimizer.LARS(
                self.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )
        elif self.args.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            raise NotImplementedError("指定されているoptimizerを利用してください。")


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
