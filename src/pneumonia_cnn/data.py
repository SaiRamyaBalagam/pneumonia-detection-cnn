"""Data loading and splitting.

Fixes the two concrete data-handling gaps found in the original project:
  1. Trains on the full class-subfolder directories the report claims to use,
     not a hand-picked 25/25 sample.
  2. The Kaggle chest_xray dataset's official val/ split is only 16 images
     (a known flaw in that dataset). We pool train/ + val/ and re-split
     stratified, keeping test/ completely untouched as the held-out set.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import Config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_pairs(directory: str, class_names: List[str]) -> List[Tuple[str, int]]:
    """Every (filepath, label_index) under directory/<class_name>/ for each class."""
    pairs: List[Tuple[str, int]] = []
    root = Path(directory)
    for idx, cls in enumerate(class_names):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTENSIONS:
                pairs.append((str(p), idx))
    return pairs


def make_corner_mask(img_size: int, frac: float) -> np.ndarray:
    """(img_size, img_size, 1) array of 1s with `frac`-sized squares zeroed
    out in all 4 corners -- removes printed laterality markers (e.g. the
    "R" marker Grad-CAM surfaced attention on) from the image before it
    reaches the model, without needing to know which corner a given image
    uses. Multiply elementwise against a normalized [0,1] image.
    """
    mask = np.ones((img_size, img_size, 1), dtype=np.float32)
    c = max(1, int(round(img_size * frac)))
    mask[:c, :c, :] = 0.0   # top-left
    mask[:c, -c:, :] = 0.0  # top-right
    mask[-c:, :c, :] = 0.0  # bottom-left
    mask[-c:, -c:, :] = 0.0  # bottom-right
    return mask


def _compute_class_weights(pairs: List[Tuple[str, int]], num_classes: int) -> Dict[int, float]:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in pairs:
        counts[label] += 1
    counts = np.maximum(counts, 1)  # avoid div-by-zero for an absent class
    total = counts.sum()
    weights = total / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def _make_tf_dataset(
    pairs: List[Tuple[str, int]],
    img_size: int,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
    mask_corners: bool = False,
    mask_corner_frac: float = 0.12,
) -> tf.data.Dataset:
    paths = [p for p, _ in pairs]
    labels = [label for _, label in pairs]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=seed, reshuffle_each_iteration=True)

    corner_mask_tensor = tf.constant(make_corner_mask(img_size, mask_corner_frac)) if mask_corners else None

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        if corner_mask_tensor is not None:
            img = img * corner_mask_tensor
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.04),  # ~+/-15 degrees
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


@dataclasses.dataclass
class Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: Optional[tf.data.Dataset]
    class_weights: Optional[Dict[int, float]]
    train_count: int
    val_count: int
    test_count: int
    test_labels: List[int]  # true labels in the same order the test dataset yields batches


def build_datasets(config: Config, seed: int) -> Datasets:
    """Pool train/+val/, stratified re-split, load held-out test/ separately."""
    train_pairs = list_pairs(config.train_dir, config.class_names)
    val_pairs = list_pairs(config.val_dir, config.class_names)
    pooled = train_pairs + val_pairs

    if not pooled:
        raise FileNotFoundError(
            f"No images found under {config.train_dir!r} or {config.val_dir!r}. "
            f"Expected one subfolder per class name in {config.class_names}."
        )

    labels = [label for _, label in pooled]
    train_split, val_split = train_test_split(
        pooled,
        test_size=config.val_fraction,
        random_state=seed,
        stratify=labels,
    )

    test_pairs = list_pairs(config.test_dir, config.class_names)

    class_weights = _compute_class_weights(train_split, config.num_classes) if config.use_class_weights else None

    mask_kwargs = {"mask_corners": config.mask_corners, "mask_corner_frac": config.mask_corner_frac}

    train_ds = _make_tf_dataset(
        train_split, config.img_size, config.batch_size, augment=True, shuffle=True, seed=seed, **mask_kwargs
    )
    val_ds = _make_tf_dataset(
        val_split, config.img_size, config.batch_size, augment=False, shuffle=False, seed=seed, **mask_kwargs
    )
    test_ds = (
        _make_tf_dataset(
            test_pairs, config.img_size, config.batch_size, augment=False, shuffle=False, seed=seed, **mask_kwargs
        )
        if test_pairs
        else None
    )

    return Datasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        class_weights=class_weights,
        train_count=len(train_split),
        val_count=len(val_split),
        test_count=len(test_pairs),
        test_labels=[label for _, label in test_pairs],
    )
