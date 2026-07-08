from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    save_json,
    save_text,
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

    checkpoint_path = Path(args.model_path)
    run_name = checkpoint_path.parent.name
    result_dir = Path(args.result_dir)
    if result_dir.name != run_name:
        result_dir = result_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

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

    correct_per_class = confusion.diag()
    total_per_class = confusion.sum(dim=1).clamp(min=1)
    per_class_accuracy = (correct_per_class.float() / total_per_class.float()).tolist()

    save_confusion_matrix(confusion, CIFAR10_CLASSES, result_dir / "confusion_matrix_eval.png")
    save_json(result_dir / "eval_summary.json", {
        "model_path": args.model_path,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_acc": float(checkpoint.get("best_acc", 0.0)),
        "seed": args.seed,
        "per_class_accuracy": {
            name: float(acc) for name, acc in zip(CIFAR10_CLASSES, per_class_accuracy)
        },
    })
    save_text(result_dir / "eval_notes.txt", "\n".join([
        f"model_path: {args.model_path}",
        f"test_loss: {test_loss:.4f}",
        f"test_acc: {test_acc * 100:.2f}%",
        f"checkpoint_epoch: {checkpoint.get('epoch')}",
        f"checkpoint_best_acc: {checkpoint.get('best_acc', 0.0) * 100:.2f}%",
        "per_class_accuracy:",
        *[f"  {name}: {acc * 100:.2f}%" for name, acc in zip(CIFAR10_CLASSES, per_class_accuracy)],
    ]))


if __name__ == "__main__":
    main()
