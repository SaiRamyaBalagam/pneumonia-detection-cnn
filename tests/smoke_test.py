"""End-to-end smoke test: proves the pipeline (data -> model -> train ->
evaluate -> gradcam) runs cleanly on this machine without needing the real
dataset, Kaggle credentials, a GPU, or any network access (pretrained=False,
so no ImageNet weight download either). Run with:

    python tests/smoke_test.py

Does NOT prove the model is accurate -- it trains on random-noise images for
1 epoch with a randomly-initialized backbone, purely to prove nothing
crashes end-to-end. Real numbers require
notebooks/colab_train_binary.ipynb on the actual dataset.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_sample_data import make_sample_data  # noqa: E402

from pneumonia_cnn.config import Config  # noqa: E402
from pneumonia_cnn.evaluate import evaluate_seed  # noqa: E402
from pneumonia_cnn.gradcam import generate_gradcam_samples  # noqa: E402
from pneumonia_cnn.train import train_one_seed  # noqa: E402


def run_scenario(label: str, data_root: Path, results_dir: Path, **config_overrides) -> None:
    print(f"\n=== Scenario: {label} ===")
    config = Config(
        class_names=["NORMAL", "PNEUMONIA"],
        data_root=str(data_root),
        img_size=96,
        batch_size=4,
        epochs=1,
        seeds=[42],
        pretrained=False,  # no network access needed for the smoke test
        results_dir=str(results_dir),
        **config_overrides,
    )

    print("--- train.py ---")
    meta = train_one_seed(config, seed=42)
    assert Path(meta["model_path"]).exists(), "Model checkpoint was not saved"
    print(f"OK: model saved to {meta['model_path']}")

    print("--- evaluate.py ---")
    result = evaluate_seed(config, seed=42)
    assert result["test_count"] == 20, f"expected 20 test images, got {result['test_count']}"
    assert 0.0 <= result["accuracy"] <= 1.0
    print(f"OK: test accuracy = {result['accuracy']:.3f} on {result['test_count']} synthetic images")

    print("--- gradcam.py ---")
    saved = generate_gradcam_samples(config, seed=42, n_per_class=1)
    assert len(saved) > 0, "No Grad-CAM overlays were saved"
    for p in saved:
        assert Path(p).exists(), f"missing {p}"
    print(f"OK: saved {len(saved)} Grad-CAM overlays")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="pneumonia_smoke_"))
    print(f"Using temp dir: {tmp_dir}")
    try:
        data_root = tmp_dir / "chest_xray"
        make_sample_data(
            root=data_root,
            class_names=["NORMAL", "PNEUMONIA"],
            counts={
                "train": {"NORMAL": 20, "PNEUMONIA": 20},
                "val": {"NORMAL": 4, "PNEUMONIA": 4},
                "test": {"NORMAL": 10, "PNEUMONIA": 10},
            },
            img_size=96,
        )

        run_scenario("baseline (frozen, unmasked)", data_root, tmp_dir / "results")
        run_scenario(
            "fine-tune + corner-masking (the two new code paths)",
            data_root,
            tmp_dir / "results_ft_masked",
            fine_tune=True,
            fine_tune_layers=4,
            fine_tune_epochs=1,
            mask_corners=True,
            mask_corner_frac=0.12,
        )

        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
