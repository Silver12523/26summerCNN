from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
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

CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def unpickle(file_path: str | Path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        with open(file_path, "rb") as handle:
            return pickle.load(handle, encoding="bytes")


class LocalCIFAR10(Dataset):
    def __init__(self, root: str | Path, train: bool, transform=None):
        self.root = Path(root)
        self.train = train
        self.transform = transform

        self.base_folder = self._resolve_base_folder()
        self.data, self.targets = self._load_data()

    def _resolve_base_folder(self) -> Path:
        candidates = [
            self.root / "cifar-10-batches-py",
            self.root / "cifar",
        ]
        for candidate in candidates:
            if (candidate / "batches.meta").exists():
                return candidate

        if (self.root / "batches.meta").exists():
            return self.root

        raise FileNotFoundError(
            f"Could not find CIFAR-10 batch files under {self.root}. "
            "Expected batches.meta and data_batch_* files."
        )

    def _load_data(self):
        batch_names = (
            [f"data_batch_{index}" for index in range(1, 6)]
            if self.train
            else ["test_batch"]
        )

        data_parts = []
        target_parts = []
        for batch_name in batch_names:
            batch = unpickle(self.base_folder / batch_name)
            data_parts.append(batch[b"data"])
            target_parts.extend(batch[b"labels"])

        data = torch.cat([torch.from_numpy(part) for part in data_parts], dim=0)
        data = data.view(-1, 3, 32, 32)
        targets = torch.tensor(target_parts, dtype=torch.long)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index].permute(1, 2, 0).byte().numpy()
        image = Image.fromarray(image)
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)


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

    data_root = Path(data_dir)
    if (data_root / "cifar-10-batches-py").exists() or (data_root / "cifar").exists() or (data_root / "batches.meta").exists():
        train_base_set = LocalCIFAR10(root=data_root, train=True, transform=train_transform)
        val_base_set = LocalCIFAR10(root=data_root, train=True, transform=test_transform)
        test_set = LocalCIFAR10(root=data_root, train=False, transform=test_transform)
    else:
        train_base_set = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        val_base_set = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=test_transform,
        )
        test_set = datasets.CIFAR10(
            root=data_root,
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
