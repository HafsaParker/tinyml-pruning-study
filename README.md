# TinyML Pruning Study

Investigating whether **layer sensitivity** can provide an objective early-warning signal for **pruning-induced accuracy collapse** in TinyML-scale convolutional neural networks, and how that warning depends on **dataset complexity** and **random seed**.

**Author:** Hafsa Parker  
**Degree:** MS Artificial Intelligence, NUST PNEC

---

# Research Question

Can the gap between the most- and least-sensitive layers of a pruned neural network reliably warn of an impending accuracy collapse?

This project investigates:

- Does an early-warning signal exist?
- Is it reproducible across random seeds?
- Does it depend on dataset complexity?
- Can it eventually be used to guide more adaptive pruning strategies?

---

# Contributions

This repository presents:

- An objective methodology for defining both **warning onset** and **collapse onset** during iterative pruning.
- Evaluation across **3 datasets × 3 independent random seeds** using one consistent methodology.
- Evidence that warning windows are **dataset-dependent** and **seed-dependent**, rather than fixed properties of the model.
- A fully reproducible experimental pipeline for studying pruning dynamics in TinyML-scale CNNs.

---

# Core Finding

> **An early-warning signal exists, but it is not a fixed property of the network. Its lead time depends on both the dataset and the random seed.**

Using one fixed methodology across all datasets and seeds:

| Dataset | Mean Warning Window | Seed Standard Deviation |
|----------|-------------------:|------------------------:|
| MNIST | 1.0 percentage points | ±2.0 |
| KMNIST | 8.7 percentage points | ±3.5 |
| FashionMNIST | 16.3 percentage points | ±8.4 |

These results suggest that **single-seed pruning studies may not reliably characterize collapse timing**, as the estimated warning window varies across random seeds.

---

## Overview of the Main Result

The figure below summarizes warning onset and collapse onset obtained using the same methodology across all datasets and random seeds.

<p align="center">
  <img src="figures/warning_collapse_dumbbell.png" width="900">
</p>

Open circles denote **warning onset**, filled circles denote **collapse onset**, and the connecting line represents the available warning window before irreversible accuracy degradation.

---

# Methodology

## Model

TinyNet CNN (~52k parameters)

```
Conv1 → Conv2 → FC1 → FC2
```

---

## Datasets

- MNIST
- FashionMNIST
- KMNIST

The same architecture and training pipeline were used for all datasets.

---

## Pruning

- Unstructured magnitude pruning
- Uniform iterative pruning
- Per-layer sensitivity analysis
- Fine-grained sparsity checkpoints
- Three independent random seeds

---

## Warning Onset

Warning onset is defined as:

> **The first pruning checkpoint where the layer-sensitivity gap exceeds a 2 percentage-point background threshold and never falls below that threshold again for the remainder of pruning.**

---

## Collapse Onset

Three candidate definitions were evaluated.

### Candidate A (Selected)

Collapse begins at the first abnormal accuracy decline that

1. exceeds the model's normal baseline variation ("wobble"), and
2. never recovers to within that same normal variation.

Normal baseline variation is estimated from a common **20–45% calibration region** and is floored at **2 percentage points**.

Candidate A was selected because it

- works consistently across datasets,
- behaves robustly across random seeds,
- and is straightforward for other researchers to reproduce.

---

### Candidate B

Change-point detection using PELT.

---

### Candidate C

Baseline-region statistical threshold.

---

Candidate A was selected as the primary methodology after comparing all three approaches.

---

## Warning Window

Warning Window = Collapse Onset − Warning Onset

This measures how much advance warning exists before irreversible accuracy degradation begins.

---

# Repository Structure

Notebooks and data files currently live flat in the repository root (folder reorganization into `notebooks/` / `figures/` / `json/` is a planned cleanup, not yet done):

```
01_baseline_training.ipynb
02_pruning_experiments.ipynb
03_FashionMNIST_experiments.ipynb
05_KMNIST_experiments.ipynb
06_FashionMNIST_multiseed.ipynb
07_MNIST_multiseed.ipynb
08_KMNIST_multiseed.ipynb
09_Warning_oset.ipynb          (warning onset / collapse onset methodology, all datasets)

mnist_full_sweep.json
fashion_gap_20_80.json
fashion_uniform_acc.json
kmnist_multiseed_results.json
kmnist_uniform_acc.json
...additional per-dataset sweep and figure files

README.md
```

**Known issues, not yet cleaned up:** `09_Warning_oset.ipynb` has a typo in its filename (should read `09_Warning_onset.ipynb`); several figure/JSON files have inconsistent capitalization or near-duplicate names from iterative saves during multi-seed development. A folder reorganization and filename cleanup pass is planned but not yet done — see Future Work.

---

# Reproducibility

All experiments use

- identical network architecture,
- identical pruning methodology,
- identical warning criterion,
- identical collapse criterion,
- identical analysis pipeline,
- three independent random seeds.

No dataset-specific or seed-specific tuning was performed.

---

# Representative Analysis

The repository contains the complete set of accuracy curves, layer sensitivity curves, and sensitivity-gap plots for every dataset and random seed.

The figure shown above summarizes the final results obtained from those analyses.

---

# Known Limitation

The common **20–45% calibration region** was chosen so that the same collapse criterion could be applied consistently across every dataset and random seed.

For datasets where early sensitivity onset begins within this interval (such as FashionMNIST), the estimated baseline variation may be slightly overestimated.

This introduces a **conservative bias**:

- collapse may be detected slightly later than its true onset,
- rather than producing premature or false-positive detections.

A rolling, backward-looking calibration window has been identified as a promising direction for future work.

---

# Claim Discipline

Every conclusion in this repository is explicitly categorized as one of the following:

- **Established** — supported by prior literature.
- **Observed** — measured directly in this work.
- **Hypothesis** — proposed but not yet validated.
- **Speculation** — potential explanation requiring further investigation.

Whenever practical, predictions are recorded before experimental results are examined. This helps distinguish hypothesis generation from hypothesis testing and reduces hindsight bias.

---

# Current Status

## Completed

- Baseline model training
- Uniform pruning experiments
- Per-layer pruning experiments
- Multi-seed validation
- Layer sensitivity analysis
- Warning onset methodology
- Collapse onset methodology

## In Progress

- Computational cost analysis (savings from stopping at warning onset vs. completing a full sweep) — requested by supervisor, not yet started
- Thesis writing
- Literature review
- Results chapter

## Future Work

- Repository cleanup: filename typo fixes, folder reorganization (`notebooks/` / `figures/` / `json/`), removal of duplicate near-identical files
- Rolling, backward-looking calibration window (see Known Limitation)
- False-positive testing of sustained-rise as a diagnostic pattern
- Adaptive warning-aware pruning
- Larger neural network architectures
- Additional datasets
- Hardware-aware TinyML deployment

---

# References

- Han et al. (2015)
- Frankle & Carbin (2019)
- Blalock et al. (2020)
- Yu et al. (2021)
- Hu, Gibson & Cano (2023)
- Pesce et al. (2026)

---

# Citation

```bibtex
Coming soon.
```
