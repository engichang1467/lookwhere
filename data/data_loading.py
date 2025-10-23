from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    PILToTensor,
    Normalize,
    CenterCrop,
    RandAugment,
    RandomHorizontalFlip,
)
from timm.data.transforms import RandomResizedCropAndInterpolation
from data.augment import ResizeSmall, new_data_aug_generator


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class ImageNetDataset(Dataset):
    def __init__(
        self, dataset, do_augment, augment_type="3aug", img_size=224
    ):
        self.dataset = dataset

        assert augment_type in [
            "3aug",
            "randaug",
        ], f"augment_type must be either 3aug or randaug, not: {augment_type}"

        if not do_augment:
            small_size = img_size
            self.transform = Compose(
                [
                    ToTensor(),
                    ResizeSmall(small_size),
                    CenterCrop((img_size, img_size)),
                    Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                ]
            )

        else:
            if augment_type == "3aug":
                first_tfl = new_data_aug_generator(
                    simple_random_crop=False, color_jitter=0.3, img_size=img_size
                )
            elif augment_type == "randaug":
                scale = (0.08, 1.0)
                interpolation = "bicubic"
                first_tfl = [
                    RandomResizedCropAndInterpolation(
                        img_size, scale=scale, interpolation=interpolation
                    ),
                    RandomHorizontalFlip(),
                    RandAugment(2, 15),
                ]

            final_tfl = [
                ToTensor(),
                Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
            self.transform = Compose(first_tfl + final_tfl)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_image = self.dataset[index]["image"].convert("RGB")
        label = self.dataset[index]["label"]
        image = self.transform(original_image)  # outputs a tensor
        return (image, label)