# CIFAR-10 CNN Classification

This folder contains a PyTorch implementation for training a CNN on CIFAR-10.

## Files

- `train.py`: train the CNN, save the best model, logs, and figures.
- `evaluate.py`: evaluate a saved model on the CIFAR-10 test set.
- `model.py`: CNN model definition.
- `data.py`: CIFAR-10 download, transforms, and dataloaders.
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
python train.py --epochs 30 --batch_size 128 --lr 0.001
```

The script downloads CIFAR-10 to `data/`, saves the best model to
`checkpoints/best_model.pth`, and writes figures/logs to `results/`.

## Quick Smoke Test

Use a small subset before handing the code to another teammate:

```bash
python train.py --epochs 2 --batch_size 64 --train_subset 1000 --val_subset 200 --test_subset 200 --num_workers 0
```

This is only for checking whether the whole pipeline runs correctly. Do not use
the resulting accuracy as the final experiment result.

## Evaluate

```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

## Suggested Experiments

```bash
python train.py --epochs 30 --lr 0.001
python train.py --epochs 30 --lr 0.0005
python train.py --epochs 30 --no_augmentation
python train.py --epochs 50 --dropout 0.3
```
