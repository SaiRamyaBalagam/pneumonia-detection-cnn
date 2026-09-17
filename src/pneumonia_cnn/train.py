"""Training entrypoint.

Usage:
    python -m pneumonia_cnn.train --config configs/binary_pneumonia.yaml
    python -m pneumonia_cnn.train --config configs/binary_pneumonia.yaml --epochs 1 --seeds 42

Trains one model per seed in config.seeds. Multi-seed by design: evaluate.py
reports mean +/- std accuracy across seeds instead of a single point
estimate, which is what the original project reported off a single run on a
34-image test set.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .config import Config
from .data import build_datasets
from .model import build_model, unfreeze_top_layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one_seed(config: Config, seed: int, epochs_override: Optional[int] = None) -> dict:
    """Phase 1: trains the dense head on a frozen backbone (always).
    Phase 2 (only if config.fine_tune): unfreezes the backbone's top
    `fine_tune_layers` layers and continues training a few more epochs at a
    much lower LR. Both phases checkpoint to the same model.keras path
    (monitor=val_loss, save_best_only=True), so the file ends up holding
    whichever phase produced the best validation loss overall.
    """
    set_seed(seed)
    datasets = build_datasets(config, seed=seed)
    model = build_model(config)

    epochs = epochs_override if epochs_override is not None else config.epochs

    seed_dir = Path(config.results_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_path = seed_dir / "model.keras"

    def make_callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path), monitor="val_loss", save_best_only=True
            ),
        ]

    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=epochs,
        class_weight=datasets.class_weights,
        callbacks=make_callbacks(),
        shuffle=False,  # data.py already shuffles train_ds; avoids a spurious Keras warning
        verbose=2,
    )
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    fine_tune_start_epoch = len(history_dict.get("loss", []))

    if config.fine_tune:
        print(f"--- Phase 2: unfreezing top {config.fine_tune_layers} backbone layers, lr={config.fine_tune_lr} ---")
        unfreeze_top_layers(model, config.fine_tune_layers, config.fine_tune_lr)
        ft_history = model.fit(
            datasets.train,
            validation_data=datasets.val,
            epochs=config.fine_tune_epochs,
            class_weight=datasets.class_weights,
            callbacks=make_callbacks(),
            shuffle=False,
            verbose=2,
        )
        for k, vals in ft_history.history.items():
            history_dict.setdefault(k, []).extend(float(v) for v in vals)

    history_path = seed_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    meta = {
        "seed": seed,
        "train_count": datasets.train_count,
        "val_count": datasets.val_count,
        "test_count": datasets.test_count,
        "class_names": config.class_names,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "fine_tune": config.fine_tune,
        "fine_tune_start_epoch": fine_tune_start_epoch if config.fine_tune else None,
        "mask_corners": config.mask_corners,
    }
    with open(seed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a config YAML (see configs/).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.epochs (useful for smoke tests).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override config.seeds.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    seeds = args.seeds if args.seeds is not None else config.seeds

    all_meta = []
    for seed in seeds:
        print(f"\n=== Training seed {seed} ===")
        meta = train_one_seed(config, seed, epochs_override=args.epochs)
        all_meta.append(meta)

    summary_path = Path(config.results_dir) / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nWrote run summary to {summary_path}")


if __name__ == "__main__":
    main()
