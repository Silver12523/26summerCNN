from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(use_augmentation: bool = True):
    train_steps = []
    if use_augmentation:
        train_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(train_steps), test_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    val_size: int = 5000,
    train_subset_size: int | None = None,
    val_subset_size: int | None = None,
    test_subset_size: int | None = None,
    use_augmentation: bool = True,
    seed: int = 42,
):
    train_transform, test_transform = build_transforms(use_augmentation)

    train_base_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_base_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = len(train_base_set) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(len(train_base_set)),
        [train_size, val_size],
        generator=generator,
    )
    train_set = Subset(train_base_set, train_subset.indices)
    val_set = Subset(val_base_set, val_subset.indices)

    if train_subset_size is not None:
        train_set = Subset(train_set, range(min(train_subset_size, len(train_set))))
    if val_subset_size is not None:
        val_set = Subset(val_set, range(min(val_subset_size, len(val_set))))
    if test_subset_size is not None:
        test_set = Subset(test_set, range(min(test_subset_size, len(test_set))))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
