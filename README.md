# CIFAR-10 CNN Classification

This folder contains a PyTorch implementation for training a CNN on CIFAR-10. It supports both local CIFAR batch files and the standard torchvision download path.

## Files

- `train.py`: train the CNN, save the best model, logs, and figures.
- `evaluate.py`: evaluate a saved model on the CIFAR-10 test set.
- `model.py`: CNN model definition.
- `data.py`: CIFAR-10 loading from local batch files or torchvision, plus transforms and dataloaders.
- `utils.py`: training, evaluation, checkpoint, and plotting utilities.
- `requirements.txt`: Python dependencies.

## Install

```bash
pip install -r requirements.txt
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch version from the
official PyTorch website if the default `pip install` does not detect CUDA.

## Train

```bash
python train.py --data_dir cifar --epochs 30 --batch_size 128 --lr 0.001 --run_name exp1
```

If you already have the extracted CIFAR-10 batch files, pass their folder with `--data_dir`. Otherwise the script downloads CIFAR-10 to `data/`, saves the best model to `checkpoints/best_model.pth`, and writes figures/logs to `results/`.

What the output means:
- `train_loss`: training set average loss for that epoch.
- `train_acc`: training set accuracy for that epoch.
- `val_loss`: validation set average loss for that epoch.
- `val_acc`: validation set accuracy for that epoch.
- `Saved best model`: the current validation accuracy is the best so far, so the model checkpoint is overwritten.

Why we split train/val/test:
- training set: used to update model weights;
- validation set: used to choose hyperparameters and save the best checkpoint;
- test set: used only once at the end, as the final unbiased report of performance.

## Quick Smoke Test

Use a small subset before handing the code to another teammate:

```bash
python train.py --data_dir cifar --epochs 2 --batch_size 64 --train_subset 1000 --val_subset 200 --test_subset 200 --num_workers 0 --run_name smoke_test
```

This is a smoke test: a fast "does the whole pipeline work?" check.
- Purpose: confirm the code can read data, start training, save checkpoints, and run evaluation without crashing.
- Use it after code changes, dependency changes, or before a long full training run.
- Do not treat its accuracy as a real result; the subset is too small and too noisy.

Recommended use:
- one quick smoke test after setup;
- then one full run for the actual report;
- then extra runs only when changing hyperparameters.

## Full Run

This is the real experiment run.

Recommended command:

```bash
python train.py --data_dir cifar --epochs 30 --batch_size 128 --lr 0.001 --run_name exp1
```

Why this is the "real" test:
- it uses the full training/validation split;
- it trains for the planned number of epochs;
- it produces the curves and checkpoint you can analyze and report.

After training, evaluate the best checkpoint:

```bash
python evaluate.py --data_dir cifar --model_path checkpoints/exp1/best_model.pth --result_dir results/exp1
```

## Evaluate

```bash
python evaluate.py --data_dir cifar --model_path checkpoints/exp1/best_model.pth --result_dir results/exp1
```

## Suggested Experiments

```bash
python train.py --epochs 30 --lr 0.001
python train.py --epochs 30 --lr 0.0005
python train.py --epochs 30 --no_augmentation
python train.py --epochs 50 --dropout 0.3
```
