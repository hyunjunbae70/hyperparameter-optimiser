import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from src.data.transforms import get_train_transforms, get_test_transforms


class CIFAR10DataLoader:
    def __init__(self, data_dir: str = "./data", val_split: float = 0.1):
        self.data_dir = data_dir
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        full_train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=get_train_transforms(),
        )

        val_size = int(len(full_train_dataset) * self.val_split)
        train_size = len(full_train_dataset) - val_size

        self.train_dataset, val_dataset_temp = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=get_test_transforms(),
        )

        val_transform = get_test_transforms()
        val_dataset_indices = val_dataset_temp.indices
        val_base_dataset = CIFAR10(
            root=self.data_dir, train=True, download=False, transform=val_transform
        )
        self.val_dataset = torch.utils.data.Subset(
            val_base_dataset, val_dataset_indices
        )

    def get_train_loader(self, batch_size: int, num_workers: int = 2) -> DataLoader:
        if self.train_dataset is None:
            self.prepare_data()

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_val_loader(self, batch_size: int, num_workers: int = 2) -> DataLoader:
        if self.val_dataset is None:
            self.prepare_data()

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_test_loader(self, batch_size: int, num_workers: int = 2) -> DataLoader:
        if self.test_dataset is None:
            self.prepare_data()

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple:
        return (3, 32, 32)
