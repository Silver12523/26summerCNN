# CIFAR-10 Experiment Results

This folder stores the experiment outputs that should be shared with teammates and included in the report. The local dataset is not included in the repository.

## What is stored here

Each experiment has its own subfolder, such as `exp1`, `exp2`, ... `exp12`.

Typical files in each experiment folder:
- `train_log.csv`: epoch-by-epoch training history.
- `loss_curve.png`: training and validation loss curve.
- `accuracy_curve.png`: training and validation accuracy curve.
- `confusion_matrix.png`: confusion matrix for the best checkpoint after training.
- `eval_summary.json`: evaluation metrics on the test set.
- `eval_notes.txt`: readable evaluation summary.
- `summary.json`: training summary and full hyperparameter record.
- `notes.txt`: short human-readable run notes.

## Experiment summary

| Exp | Main change | Best Val Acc | Test Acc | Short note |
|---|---|---:|---:|---|
| exp1 | Baseline | 86.60% | 85.76% | Starting point |
| exp2 | `lr=0.0005` | 86.04% | 85.85% | Similar to baseline |
| exp3 | No augmentation | 84.58% | 84.04% | Clear drop |
| exp4 | `dropout=0.3` | 87.62% | 86.65% | Better than baseline |
| exp5 | `weight_decay=5e-4` | 87.04% | 86.41% | Slight gain, not best |
| exp6 | `batch_size=64` | 87.22% | 86.87% | Small improvement |
| exp7 | `epochs=50` | 87.98% | 87.53% | More training helps |
| exp8 | `batch_size=64, dropout=0.3, epochs=50` | 89.86% | 88.49% | Strong improvement |
| exp9 | `dropout=0.2` | 89.18% | 88.60% | Good generalization |
| exp10 | `dropout=0.3, weight_decay=5e-5` | 88.66% | 88.70% | Best test among this group |
| exp11 | `dropout=0.4` | 89.68% | 88.49% | Strong val, similar test |
| exp12 | `dropout=0.2, weight_decay=5e-5` | 89.90% | 88.73% | Best overall so far |
| exp13 | `dropout=0.15, weight_decay=5e-5` | 88.78% | 88.94% | Best test so far |

## Main conclusions

- Learning rate change from `0.001` to `0.0005` does not matter much.
- Data augmentation is important for CIFAR-10.
- The best region seems to be around `dropout=0.15~0.3` and `weight_decay=5e-5~1e-4`.
- `batch_size=64` is slightly better than `128`.
- Training longer than 30 epochs helps, and 50 epochs is better for this model.

## Handoff notes

- The dataset is not committed. The local CIFAR folder should stay outside version control.
- If a teammate only needs analysis, the experiment folders in `results/` are enough.
- If someone needs to reproduce or continue training, the matching checkpoint in `checkpoints/` is useful, but not strictly required for pure analysis.
- Recommended next direction: a narrower search around `dropout=0.15~0.25` and `weight_decay=5e-5~1e-4`, or move to model-architecture changes.
