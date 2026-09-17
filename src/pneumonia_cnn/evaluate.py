"""Evaluation: confusion matrix, per-class metrics, ROC-AUC, multi-seed
aggregation, and a bootstrap confidence interval on test accuracy.

Directly answers the original project's core rigor gap: a single 34-image
test run reported as if it were a stable 98% accuracy. Here we evaluate on
the full held-out test/ split (624 images for the binary Kaggle dataset --
about 18x larger) and report multi-seed mean +/- std plus a bootstrap CI
instead of one point estimate.

Usage:
    python -m pneumonia_cnn.evaluate --config configs/binary_pneumonia.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .config import Config
from .data import build_datasets


def bootstrap_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 2000, seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (y_true == y_pred).astype(np.float64)
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        accs[i] = correct[idx].mean()
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return {"mean": float(correct.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def evaluate_seed(config: Config, seed: int) -> Dict:
    seed_dir = Path(config.results_dir) / f"seed_{seed}"
    model_path = seed_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}; run train.py for seed {seed} first.")

    model = tf.keras.models.load_model(model_path)
    datasets = build_datasets(config, seed=seed)
    if datasets.test is None or datasets.test_count == 0:
        raise FileNotFoundError(f"No test images found under {config.test_dir}.")

    probs = model.predict(datasets.test, verbose=0)
    y_true = np.array(datasets.test_labels)
    y_pred = probs.argmax(axis=1)

    report = classification_report(
        y_true, y_pred, target_names=config.class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if config.num_classes == 2:
        roc_auc = float(roc_auc_score(y_true, probs[:, 1]))

    ci = bootstrap_accuracy_ci(y_true, y_pred)

    result = {
        "seed": seed,
        "test_count": datasets.test_count,
        "accuracy": ci["mean"],
        "accuracy_ci_95": [ci["ci_low"], ci["ci_high"]],
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(seed_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def aggregate(results: List[Dict]) -> Dict:
    accs = np.array([r["accuracy"] for r in results])
    return {
        "n_seeds": len(results),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "per_seed_accuracy": accs.tolist(),
        "per_seed_ci_95": [r["accuracy_ci_95"] for r in results],
        "test_count": results[0]["test_count"] if results else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    seeds = args.seeds if args.seeds is not None else config.seeds

    results = [evaluate_seed(config, seed) for seed in seeds]
    summary = aggregate(results)

    summary_path = Path(config.results_dir) / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
