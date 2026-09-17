"""Generates a tiny synthetic dataset with the same directory layout data.py
expects (train/<class>/, val/<class>/, test/<class>/), so the pipeline can be
smoke-tested end-to-end without downloading anything.

Images are random noise, not real X-rays -- this only proves the code runs,
not that the model learns anything meaningful. Real numbers come from
notebooks/colab_train_binary.ipynb on the actual Kaggle dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def make_sample_data(
    root: Path,
    class_names: List[str],
    counts: Dict[str, Dict[str, int]],
    img_size: int = 96,
    seed: int = 0,
) -> None:
    """counts example: {"train": {"NORMAL": 20, "PNEUMONIA": 20}, "val": {...}, "test": {...}}"""
    rng = np.random.default_rng(seed)
    for split, per_class in counts.items():
        for cls in class_names:
            n = per_class.get(cls, 0)
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
                Image.fromarray(arr).save(cls_dir / f"{cls.lower()}_{i:03d}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    make_sample_data(
        root=Path(args.root),
        class_names=["NORMAL", "PNEUMONIA"],
        counts={
            "train": {"NORMAL": 20, "PNEUMONIA": 20},
            "val": {"NORMAL": 4, "PNEUMONIA": 4},  # deliberately tiny, like the real Kaggle val/
            "test": {"NORMAL": 10, "PNEUMONIA": 10},
        },
    )
    print(f"Wrote synthetic dataset under {args.root}")
