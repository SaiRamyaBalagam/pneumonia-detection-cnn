"""Grad-CAM explainability.

Cheap, high-credibility addition: shows whether the model is actually
looking at lung regions or at scan artifacts/markers (a very common failure
mode in chest X-ray classifiers, and a standard reviewer question for
medical-imaging papers). The original project had no explainability step at
all.

Usage:
    python -m pneumonia_cnn.gradcam --config configs/binary_pneumonia.yaml --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

from .config import Config
from .data import list_pairs, make_corner_mask
from .model import BACKBONE_SUBMODEL_NAME

LAST_CONV_LAYER = {
    "vgg16": "block5_conv3",
}


def load_image(path: str, img_size: int, mask_corners: bool = False, mask_corner_frac: float = 0.12) -> np.ndarray:
    """Must match data.py's preprocessing exactly (including corner masking,
    if the model being explained was trained with it) -- otherwise the
    model sees a different input distribution here than it saw in training,
    and the resulting heatmap wouldn't reflect real model behavior.
    """
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if mask_corners:
        arr = arr * make_corner_mask(img_size, mask_corner_frac)
    return arr


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    backbone_name: str,
    pred_index: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Returns (heatmap in [0, 1] shaped (h, w), the predicted/target class index).

    Nested-model gotcha: `conv_layer.output` belongs to the backbone
    submodel's own original graph, so we can't wire it directly to the
    outer model's output tensor (those come from a *different* call of the
    backbone, inside the outer functional graph). Instead we build a model
    from base.input to (conv_output, base_output), then manually re-apply
    the head layers (GAP/Dense/Dropout/Dense) to base_output inside the same
    GradientTape, which keeps everything on one differentiable graph.
    """
    backbone = model.get_layer(BACKBONE_SUBMODEL_NAME)
    conv_layer = backbone.get_layer(LAST_CONV_LAYER[backbone_name])
    grad_model = tf.keras.Model(inputs=backbone.input, outputs=[conv_layer.output, backbone.output])

    inputs = tf.expand_dims(tf.convert_to_tensor(img_array), axis=0)

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(inputs)
        tape.watch(conv_output)
        x = base_output
        for layer in model.layers:
            if layer.name == BACKBONE_SUBMODEL_NAME or isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def overlay_heatmap(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    h, w = img_array.shape[:2]
    heatmap_resized = np.asarray(Image.fromarray(np.uint8(255 * heatmap)).resize((w, h)))
    colored = cm.jet(heatmap_resized / 255.0)[:, :, :3]
    overlaid = np.clip((1 - alpha) * img_array + alpha * colored, 0, 1)
    return Image.fromarray(np.uint8(overlaid * 255))


def generate_gradcam_samples(config: Config, seed: int, n_per_class: int = 2, rng_seed: int = 0) -> List[str]:
    """Saves Grad-CAM overlays for up to n_per_class correct and n_per_class
    incorrect test predictions per class into results/seed_<seed>/gradcam/.
    Returns the list of saved file paths.
    """
    seed_dir = Path(config.results_dir) / f"seed_{seed}"
    model_path = seed_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}; run train.py for seed {seed} first.")

    model = tf.keras.models.load_model(model_path)
    test_pairs = list_pairs(config.test_dir, config.class_names)
    if not test_pairs:
        raise FileNotFoundError(f"No test images found under {config.test_dir}.")

    rng = random.Random(rng_seed)
    rng.shuffle(test_pairs)

    out_dir = seed_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    # bucket by (true_label, correct/incorrect) so we can cap n_per_class each
    buckets = {(t, correct): [] for t in range(config.num_classes) for correct in (True, False)}

    for path, true_label in test_pairs:
        img_array = load_image(path, config.img_size, config.mask_corners, config.mask_corner_frac)
        probs = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        pred_label = int(np.argmax(probs))
        key = (true_label, pred_label == true_label)
        if len(buckets[key]) >= n_per_class:
            continue
        buckets[key].append((path, img_array, true_label, pred_label, float(probs[pred_label])))

        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    for (true_label, correct), items in buckets.items():
        for path, img_array, tl, pred_label, prob in items:
            heatmap, _ = make_gradcam_heatmap(model, img_array, config.backbone, pred_index=pred_label)
            overlay = overlay_heatmap(img_array, heatmap)
            tag = "correct" if correct else "incorrect"
            fname = (
                f"{config.class_names[tl]}_pred-{config.class_names[pred_label]}"
                f"_{tag}_{Path(path).stem}.png"
            )
            out_path = out_dir / fname
            overlay.save(out_path)
            saved.append(str(out_path))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-per-class", type=int, default=2)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    saved = generate_gradcam_samples(config, args.seed, n_per_class=args.n_per_class)
    print(f"Saved {len(saved)} Grad-CAM overlays:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
