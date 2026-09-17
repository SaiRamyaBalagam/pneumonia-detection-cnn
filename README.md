# Pneumonia Chest X-Ray Classifier -- rigorous rebuild

A rebuild of a UNT course project that fixes a gap between what the
project's report claimed and what its code actually did, and restructures
the code into something that can support real follow-on research (rather
than a single notebook that only runs once).

## What was wrong with the original

The original report/abstract claimed a model trained on the full **5,863-image
Kermany chest X-ray dataset** (Normal vs Pneumonia). The actual notebook
never did that -- it built an ad hoc 44-image dataset (25 COVID X-rays +
25 sampled "normal" images) and reported 97-98% accuracy off a **34-image**
test set. With `n≈5` in the minority test class, a single misclassification
swings the reported accuracy by several points; the number wasn't wrong, it
was just not meaningful at that scale.

| | Original notebook | This rebuild |
|---|---|---|
| Task | Framed as pneumonia detection, actually trained COVID vs Normal | Normal vs Pneumonia (matches the report's actual claim) |
| Train images | ~44 (hand-picked) | ~4,700 (90% of pooled train+val) |
| Val images | none (no real validation split) | ~520 (proper stratified 10% held out) |
| Test images | 34 | 624 (official held-out Kaggle test set, untouched) |
| Runs reported | 1 | 3 seeds, mean +/- std + bootstrap 95% CI |
| Explainability | none | Grad-CAM overlays on sample predictions |
| Class imbalance handling | none | class-weighted loss |
| **Reported accuracy** | **97-98%** (single run, n=34, not reproducible) | **86.9% ± 0.8%** (3 seeds, n=624) |

## Results (measured on Colab, real dataset)

```json
{
  "n_seeds": 3,
  "accuracy_mean": 0.8691,
  "accuracy_std": 0.0079,
  "per_seed_accuracy": [0.8782, 0.8654, 0.8638],
  "per_seed_ci_95": [[0.8526, 0.9039], [0.8365, 0.8911], [0.8365, 0.8910]],
  "test_count": 624
}
```

**86.9% ± 0.8% on the full 624-image held-out test set** -- numerically
lower than the original's claimed 97-98%, but far more trustworthy: three
independently-initialized models, each with its own random train/val split,
landed within 1.5 points of each other. That tight spread is itself evidence
this is a real, reproducible signal rather than one lucky split on a
34-image sample.

Why lower than the original's claim, honestly:
- The original's number came from a much easier, different task (COVID vs.
  Normal, visually more distinct) on 34 hand-picked images, not from Normal
  vs. Pneumonia at scale -- the two numbers were never actually comparable.
- This rebuild trains a linear head on a **frozen** VGG16 backbone (no
  fine-tuning of the convolutional layers) in the baseline config. Tested
  as an ablation below -- fine-tuning turned out *not* to reliably improve
  on this, so the frozen baseline isn't leaving obvious accuracy on the
  table after all.
- The official Kaggle `test/` split for this dataset is known to be a
  somewhat harder, slightly out-of-distribution sample relative to
  `train/` -- other published work on this exact dataset commonly lands
  in the 85-92% range without fine-tuning, and higher with it.

**Grad-CAM finding**: reviewing the overlay grid surfaced a real, literature-
consistent failure mode: several test images (both correctly and
incorrectly classified) show the model's strongest activation on the "R"
laterality marker printed in the image corner rather than on lung tissue --
e.g. `person48_virus_100` (incorrect) and `person157_bacteria_740`
(correct). This is a known shortcut-learning risk in chest X-ray deep
learning (models keying on scanner/annotation artifacts that happen to
correlate with the label -- see Zech et al. 2018, "Confounding variables can
degrade generalization performance of radiological deep learning models").
Tested directly as an ablation below -- masking those corners out cost a
real, measurable amount of accuracy (see "Ablation study"), so the model's
performance does appear to depend on something in that region, though not
necessarily the marker text specifically.

## Ablation study: fine-tuning and corner-masking

Two follow-ups from the baseline result above, each a runnable config
(`pneumonia_cnn/model.py`'s `unfreeze_top_layers()` and `pneumonia_cnn/
data.py`'s `make_corner_mask()`), all now confirmed at 3 seeds. The baseline
and fine-tune configs were each independently run twice (two full 3-seed
batches, 6 seeds total per variant) -- useful as a reproducibility check,
since GPU training isn't perfectly deterministic even with a fixed seed.

| Config | Frozen/fine-tuned | Corners masked | Accuracy (3 seeds) |
|---|---|---|---|
| `binary_pneumonia.yaml` -- run 1 | frozen | no | 86.9% ± 0.8% (87.8, 86.5, 86.4) |
| `binary_pneumonia.yaml` -- run 2 (replication) | frozen | no | 86.4% ± 1.5% (88.0, 86.1, 85.1) |
| `binary_pneumonia_finetune.yaml` -- run 1 | fine-tuned (top 4 layers, 6 epochs @ 1e-5) | no | 86.4% ± 3.0% (87.5, 88.8, 83.0) |
| `binary_pneumonia_finetune.yaml` -- run 2 (replication) | fine-tuned | no | 86.2% ± 2.2% (86.9, 88.0, 83.8) |
| `binary_pneumonia_masked.yaml` | frozen | yes (12% corners) | **84.9% ± 2.2%** (87.0, 84.9, 82.7) |
| `binary_pneumonia_finetune_masked.yaml` | fine-tuned | yes | **85.5% ± 5.6%** (87.5, 89.7, 79.2) |

**Baseline replicates well**: two independent 3-seed runs landed at 86.9%
and 86.4% -- consistent with each other, giving a pooled estimate around
86.6% for the frozen backbone's true performance on this test set.

**Fine-tuning: confirmed twice over, and the conclusion held both times.** A
single seed 42 run initially showed 89.1% -- above the baseline's entire
range, and looked like a clear win. Both full 3-seed runs told a different
story: mean accuracy (86.4%, then 86.2% on replication) is essentially
unchanged from baseline, but the seed-to-seed **standard deviation is 2-4x
higher** than baseline's (3.0 and 2.2 points vs. 0.8-1.5). The honest,
now well-replicated conclusion: **unfreezing the top backbone layers does
not reliably improve accuracy on this dataset, and makes training
noticeably less stable.** The original 89.1% single-seed result was an
optimistic outlier -- a concrete, measured example of why the single-seed
"directional check" protocol exists, and why it matters not to stop at the
first number.

**Corner-masking: the single-seed check was wrong, and this is a real
correction, not just an update.** The original single-seed run (86.9%,
matching baseline) suggested masking the "R" marker corner changed nothing.
**The 3-seed result contradicts that**: 84.9% ± 2.2%, roughly 1.5-2 points
below the baseline's pooled ~86.6% mean. The likely explanation isn't that
the model specifically needs the marker text -- it's that a 12%-of-image
corner mask also clips some genuine anatomy near the image edges (upper lung
apex, clavicle), so removing it costs a modest amount of real signal along
with the marker. This doesn't fully resolve the shortcut-learning question
Grad-CAM raised (that would need a mask shaped to the marker specifically,
not full corners), but it does mean the earlier "no effect" claim was
incorrect and has been corrected here rather than left standing.

**Fine-tune + masked combined is the least stable variant by far**: 85.5% ±
5.6%, with one seed as low as 79.2% -- a 10.6-point spread across just 3
seeds. It inherits fine-tuning's instability and stacks the masked
variant's modest accuracy cost on top.

**Bottom line across all six runs**: nothing beat the simple frozen,
unmasked baseline in a way that survived multi-seed scrutiny. Fine-tuning
adds instability without benefit; corner-masking costs a small amount of
real accuracy rather than being neutral; combining both is actively the
worst, least reliable option. That's a legitimate, useful finding on its
own -- it says this small-dataset transfer-learning setup with a frozen
VGG16 backbone is already close to what this specific approach can extract,
and that both single-seed checks in this study (fine-tuning looking better
than it was, masking looking more neutral than it was) needed the full
3-seed confirmation to get right -- in *opposite* directions, which is a
useful illustration that single-seed checks can mislead either way, not
just optimistically.

**Bottom line across all four variants: nothing beat the simple frozen
baseline in a way that survived multi-seed scrutiny.** That's a legitimate,
useful finding to report as-is, not a failure to find something -- it says
this small-dataset, transfer-learning setup is already close to what this
architecture can extract without a fundamentally different approach (more
data, a different backbone, or a different training regimen entirely).

Run via `notebooks/colab_train_binary.ipynb`'s Step 7, or locally:
```
python -m pneumonia_cnn.train --config configs/binary_pneumonia_finetune.yaml --seeds 42 43 44
python -m pneumonia_cnn.evaluate --config configs/binary_pneumonia_finetune.yaml --seeds 42 43 44
```
(same pattern for `_masked` and `_finetune_masked`).

## Repository structure

```
rebuild/
  src/pneumonia_cnn/
    config.py       # dataclass config -- multiclass later = new YAML, no code changes
    data.py          # loads chest_xray/{train,val,test}, pools+re-splits train/val, class weights,
                      # optional corner-masking ablation (make_corner_mask())
    model.py         # VGG16 frozen backbone + dense head, optional fine-tune phase
                      # (unfreeze_top_layers()); backbone registry for future comparisons
    train.py         # multi-seed, two-phase (frozen -> optional fine-tune) training CLI
    evaluate.py       # confusion matrix, per-class P/R/F1, ROC-AUC, bootstrap CI, seed aggregation
    gradcam.py         # Grad-CAM heatmaps + overlay images (masking-aware)
    federated/          # NOT implemented -- documented interface for the federated-learning
                         # extension described below (see partition.py's docstring)
  configs/
    binary_pneumonia.yaml                    # baseline -- what's been run and reported (86.9% ± 0.8%)
    binary_pneumonia_finetune.yaml            # ablation: fine-tune top 4 layers, unmasked
    binary_pneumonia_masked.yaml               # ablation: frozen, corners masked
    binary_pneumonia_finetune_masked.yaml       # ablation: both combined
    multiclass_template.yaml                    # documented placeholder for the multiclass extension
  tests/
    make_sample_data.py         # synthetic image generator, no internet needed
    smoke_test.py                 # proves the pipeline runs end-to-end (data->train->eval->gradcam)
                                   # on synthetic data with a randomly-initialized backbone --
                                   # this is a correctness check, not an accuracy claim
  notebooks/
    colab_train_binary.ipynb    # self-contained: pulls the real Kaggle dataset, trains on GPU,
                                  # evaluates, runs Grad-CAM, packages results for download
  results/    # gitignored -- populated by train.py / evaluate.py / gradcam.py
```

## Running it

**Locally (correctness check only, no GPU/internet needed):**
```
pip install -r requirements.txt
python tests/smoke_test.py
```
This trains on synthetic random-noise images with `pretrained: false` (no
ImageNet weight download) for one epoch, purely to prove the pipeline runs
without errors. It says nothing about real accuracy.

**On Colab (real numbers, on the actual 5,863-image dataset):**
Open `notebooks/colab_train_binary.ipynb` in Google Colab, select a GPU
runtime, and run the cells top to bottom. You'll need your own Kaggle API
token (Settings -> API Tokens -> Generate New Token on kaggle.com) -- the
notebook prompts for it with a masked input so it's never written into the
notebook's saved output. Expect roughly 10-20 minutes per seed.

**Locally with a real dataset (if you have a GPU machine):**
```
python -m pneumonia_cnn.train --config configs/binary_pneumonia.yaml
python -m pneumonia_cnn.evaluate --config configs/binary_pneumonia.yaml
python -m pneumonia_cnn.gradcam --config configs/binary_pneumonia.yaml --seed 42
```
(edit `configs/binary_pneumonia.yaml`'s `data_root` to point at your local
`chest_xray/` folder first)

## Future research roadmap

This structure exists specifically so these are extensions, not rewrites:

1. **Multiclass** (Normal / Bacterial / Viral / COVID) -- `configs/
   multiclass_template.yaml` documents the data-prep steps needed
   (splitting the PNEUMONIA folder by the Kermany filename convention,
   merging in the `covid-chestxray-dataset`). `model.py`/`train.py`/
   `evaluate.py`/`gradcam.py` already support `num_classes > 2` unchanged.
2. **Federated learning** -- `pneumonia_cnn/federated/partition.py`
   documents the intended design: simulate non-IID "hospital" clients via
   Dirichlet partitioning, train with FedAvg (Flower), compare global test
   accuracy against this rebuild's centralized baseline. This is the
   highest-value next step -- it's the motivation the original report's own
   citations (Kundu 2021) gestured at but never built.
3. **Cross-institution generalization** -- evaluate a trained
   `results/seed_*/model.keras` on a *different* source dataset (e.g. RSNA
   Pneumonia Detection Challenge) without retraining, to test whether
   accuracy survives outside the single-center (Guangzhou) source data.
   Chest X-ray models are known to sometimes key on scanner/hospital
   artifacts rather than pathology -- Grad-CAM overlays from this rebuild
   are a starting point for investigating that here too.
4. **Uncertainty quantification** -- Monte Carlo dropout or deep ensembles
   (the 3-seed setup already here is a first step toward an ensemble) to
   give calibrated confidence, framed around "when should this model defer
   to a radiologist."
