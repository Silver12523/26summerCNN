from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

# Work around a common Windows/Anaconda OpenMP duplicate runtime issue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim

from data import CIFAR10_CLASSES, get_dataloaders
from model import SimpleCNN
from utils import (
    evaluate,
    get_device,
    plot_history,
    save_checkpoint,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    save_text,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--val_subset", type=int, default=None)
    parser.add_argument("--test_subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    result_dir = Path(args.result_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        train_subset_size=args.train_subset,
        val_subset_size=args.val_subset,
        test_subset_size=args.test_subset,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
    )

    model = SimpleCNN(num_classes=10, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acc = 0.0
    best_model_path = checkpoint_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_model_path, model, optimizer, epoch, best_val_acc, args)
            print(f"Saved best model: {best_model_path} ({best_val_acc * 100:.2f}%)")

    save_history_csv(history, result_dir / "train_log.csv")
    plot_history(history, result_dir)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion = evaluate(model, test_loader, criterion, device)
    print(f"Final test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%")

    save_confusion_matrix(confusion, CIFAR10_CLASSES, result_dir / "confusion_matrix.png")
    save_json(result_dir / "summary.json", {
        "run_name": run_name,
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "checkpoint_path": str(best_model_path),
        "result_dir": str(result_dir),
        "args": {key: value for key, value in vars(args).items()},
    })
    save_text(result_dir / "notes.txt", "\n".join([
        f"run_name: {run_name}",
        f"best_epoch: {checkpoint['epoch']}",
        f"best_val_acc: {best_val_acc * 100:.2f}%",
        f"test_acc: {test_acc * 100:.2f}%",
        f"checkpoint_path: {best_model_path}",
        f"result_dir: {result_dir}",
        f"command: python train.py --data_dir {args.data_dir} --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --run_name {run_name}",
    ]))
    print(f"Training log and figures saved to: {result_dir}")


if __name__ == "__main__":
    main()
