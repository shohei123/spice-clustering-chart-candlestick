# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as torchvision_transforms

# MoCo
import moco.transforms as moco_transforms


class ChartDataLoader(pl.LightningDataModule):
    def init__(
        self,
        batch_size: int,
        crop_min: float,
        data_dir: str,
        persistent: bool,
        workers: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.crop_min = crop_min
        self.data_dir = data_dir
        self.persistent = persistent
        self.workers = workers

        # 画像の加工処理群の取得
        self.aug1, self.aug2 = moco_transforms.get_augmentations(
            crop_min=crop_min
        )

    def prepare_data(self):
        # 「加工された2枚の画像」がセットで格納されたデータセット
        self.train_dataset = datasets.ImageFolder(
            self.data_dir,
            moco_transforms.TwoCropsTransform(
                torchvision_transforms.Compose(self.aug1),
                torchvision_transforms.Compose(self.aug2)
            )
        )

    def train_dataloader(self):
        train_loader = torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.persistent,
            pin_memory=True,  # メモリ領域がページング（分割）されなくなり、処理の高速化が期待できる
            drop_last=True
        )

        return train_loader
