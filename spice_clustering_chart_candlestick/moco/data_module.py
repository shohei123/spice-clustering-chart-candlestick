# PyTorch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as torchvision_transforms

# Pytorch Lightning
import pytorch_lightning as pl

# MoCo
import moco.transforms as moco_transforms


class ChartDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # 画像の加工処理群の取得
        self.aug1, self.aug2 = moco_transforms.get_augmentations(
            self.args
        )

    def setup(self, stage):
        # 「加工された2枚の画像」がセットで格納されたデータセット
        self.train_dataset = datasets.ImageFolder(
            self.args.data_dir,
            moco_transforms.TwoCropsTransform(
                torchvision_transforms.Compose(self.aug1),
                torchvision_transforms.Compose(self.aug2)
            )
        )

    def train_dataloader(self):
        train_loader = torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers=self.args.persistent,
            pin_memory=True,  # メモリ領域がページング（分割）されなくなり、処理の高速化が期待できる
            drop_last=True
        )

        # バッチ数を取得してargsに格納
        self.args.num_batches = len(train_loader)

        return train_loader
