from PIL import ImageFilter, ImageOps
import random
import torchvision.transforms as transforms


class TwoCropsTransform:
    """1枚の画像からランダムに2枚切り出す"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """
    Gaussian blur augmentation from SimCLR:
    https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# ピクセル反転
class Solarize(object):
    """
    Solarize augmentation from BYOL:
    https://arxiv.org/abs/2006.07733
    """

    def __call__(self, x):
        return ImageOps.solarize(x)


def get_augmentations(crop_min: float):
    # 画像データの加工処理郡
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 画像加工についての参考論文: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    return augmentation1, augmentation2
