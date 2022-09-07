# PyTorch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as torchvision_transforms

# MoCo
import moco.transforms as moco_transforms


def ChartDataLoader(
        crop_min,
        batch_size: int = 32,
        data_dir: str = "../datasets/img",
        persistent: bool = True,
        workers: int = 1
):

    # 画像の加工処理群の取得
    aug1, aug2 = moco_transforms.get_augmentations(
        crop_min=crop_min
    )

    # 「加工された2枚の画像」がセットで格納されたデータセット
    train_dataset = datasets.ImageFolder(
        data_dir,
        moco_transforms.TwoCropsTransform(
            torchvision_transforms.Compose(aug1),
            torchvision_transforms.Compose(aug2)
        )
    )

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=persistent,
        pin_memory=True,  # メモリ領域がページング（分割）されなくなり、処理の高速化が期待できる
        drop_last=True
    )

    return train_loader
