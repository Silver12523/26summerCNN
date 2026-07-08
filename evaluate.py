from __future__ import annotations

import argparse
import os

# Work around a common Windows/Anaconda OpenMP duplicate runtime issue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from data import CIFAR10_CLASSES, get_dataloaders
from model import SimpleCNN
from utils import (
    evaluate,
    get_device,
    print_class_accuracy,
    save_confusion_matrix,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 CNN.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_subset", type=int, default=None)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_subset_size=args.test_subset,
        use_augmentation=False,
        seed=args.seed,
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint.get("args", {})
    dropout = model_args.get("dropout", 0.5)

    model = SimpleCNN(num_classes=10, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, confusion = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print_class_accuracy(confusion, CIFAR10_CLASSES)

    save_confusion_matrix(
        confusion,
        CIFAR10_CLASSES,
        f"{args.result_dir}/confusion_matrix_eval.png",
    )


if __name__ == "__main__":
    main()
