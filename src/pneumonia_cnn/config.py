"""Central config for the training pipeline.

Everything that differs between the binary run (this pass) and a future
multiclass run (Normal/Bacterial/Viral/COVID) lives here. Switching tasks
later should mean editing a YAML file, not touching data.py/model.py/train.py.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

import yaml


@dataclasses.dataclass
class Config:
    # --- task definition ---
    class_names: List[str]              # e.g. ["NORMAL", "PNEUMONIA"]
    data_root: str                      # dir containing train/, val/, test/ subdirs,
                                         # each with one subfolder per class name

    # --- image / model ---
    img_size: int = 224
    backbone: str = "vgg16"             # vgg16 today; model.py can add more
    dense_units: int = 64
    dropout: float = 0.5
    pretrained: bool = True             # False = random init, no ImageNet weight
                                         # download (used by the offline smoke test)

    # --- training ---
    batch_size: int = 32
    epochs: int = 15
    init_lr: float = 1e-3
    val_fraction: float = 0.10          # fraction of pooled train+val held for validation
    seeds: List[int] = dataclasses.field(default_factory=lambda: [42, 43, 44])
    use_class_weights: bool = True
    early_stopping_patience: int = 4

    # --- fine-tuning (phase 2, optional) ---
    # Phase 1 always trains only the dense head on a frozen backbone. If
    # fine_tune is set, phase 2 unfreezes the backbone's last fine_tune_layers
    # layers and continues training at a much lower LR. See model.py's
    # unfreeze_top_layers() and train.py's two-phase train_one_seed().
    fine_tune: bool = False
    fine_tune_layers: int = 4           # trailing backbone layers to unfreeze (~ last conv block for VGG16)
    fine_tune_epochs: int = 6
    fine_tune_lr: float = 1e-5

    # --- corner-masking ablation (optional) ---
    # Zeroes out corner squares of the image before it ever reaches the
    # model, to test whether accuracy depends on printed laterality markers
    # (e.g. the "R" marker Grad-CAM surfaced attention on) rather than lung
    # tissue. Applied identically to train/val/test when enabled, so the
    # comparison against an unmasked run is apples-to-apples. See data.py's
    # make_corner_mask().
    mask_corners: bool = False
    mask_corner_frac: float = 0.12      # fraction of image width/height masked in each of the 4 corners

    # --- output ---
    results_dir: str = "results"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @property
    def train_dir(self) -> str:
        return str(Path(self.data_root) / "train")

    @property
    def val_dir(self) -> str:
        """The dataset's official val/ split -- pooled into train, not used standalone.

        The Kaggle chest_xray dataset ships only 16 images here, which is why
        the original project's rigor was thin. data.py pools this into the
        train pool and re-splits properly instead of trusting it as-is.
        """
        return str(Path(self.data_root) / "val")

    @property
    def test_dir(self) -> str:
        return str(Path(self.data_root) / "test")
