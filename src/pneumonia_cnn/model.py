"""Model architecture.

Same family the original report claimed (VGG16 transfer learning, frozen
backbone + small dense head), but registered so a future backbone comparison
(EfficientNet, ConvNeXt, ViT, etc. -- one of the flagged future-research
directions) is a one-line addition to _BACKBONES, not a rewrite.
"""
from __future__ import annotations

import tensorflow as tf

from .config import Config

_BACKBONES = {
    "vgg16": tf.keras.applications.VGG16,
}

# Name assigned to the backbone submodel so gradcam.py can reliably find it
# via model.get_layer(BACKBONE_SUBMODEL_NAME) regardless of which backbone
# is configured.
BACKBONE_SUBMODEL_NAME = "backbone"


def build_model(config: Config) -> tf.keras.Model:
    if config.backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone {config.backbone!r}. Available: {list(_BACKBONES)}")

    backbone_fn = _BACKBONES[config.backbone]
    base = backbone_fn(
        weights="imagenet" if config.pretrained else None,
        include_top=False,
        input_shape=(config.img_size, config.img_size, 3),
        name=BACKBONE_SUBMODEL_NAME,
    )
    base.trainable = False  # frozen feature extractor -- matches the original report's approach

    inputs = tf.keras.Input(shape=(config.img_size, config.img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.dropout)(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{config.backbone}_pneumonia_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.init_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int, learning_rate: float) -> None:
    """Unfreezes the last `n_layers` of the backbone submodel (always named
    BACKBONE_SUBMODEL_NAME by build_model(), regardless of which backbone is
    configured) for phase-2 fine-tuning, and recompiles the model at
    (typically much lower) `learning_rate`. Call once, after phase-1
    frozen-backbone training, then continue model.fit() for a few more
    epochs -- see train.py.

    Caveat for future backbones with BatchNorm (VGG16 has none, so this
    doesn't bite yet): build_model() calls the backbone with
    training=False, which is baked into the functional graph and stays in
    effect even after unfreezing here -- harmless for VGG16 (no BatchNorm/
    Dropout inside it), but would keep BatchNorm running-stats frozen during
    fine-tuning for a backbone that has them. Revisit this if _BACKBONES
    grows to include one (ResNet, EfficientNet, ...).
    """
    backbone = model.get_layer(BACKBONE_SUBMODEL_NAME)
    backbone.trainable = True
    for layer in backbone.layers[: max(0, len(backbone.layers) - n_layers)]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
