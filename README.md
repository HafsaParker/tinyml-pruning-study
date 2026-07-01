# TinyML Pruning Study: Layer Sensitivity Analysis for Efficient Model Compression

**Author:** Hafsa Parker  
**Type:** Independent Research  
**Status:** Experiments Complete — Write-up in Progress

---

## Overview

This project presents an empirical study on neural network pruning for TinyML scale networks. Starting from Han et al. 2015 — the foundational magnitude pruning paper — this study goes beyond reproduction to investigate **where pruning tolerance breaks down, why it breaks down there, and how sensitivity-aware pruning can recover accuracy.**

The central question:

> *Can layer sensitivity profiles predict accuracy cliff location in TinyML-scale networks, and does dataset complexity govern this relationship?*

---

## Key Findings

### Finding 1 — Accuracy Cliff at 66–67% Sparsity (MNIST)
Uniform magnitude pruning maintains accuracy until a critical threshold. For MNIST, accuracy stays above 95% up to 65% sparsity, then collapses sharply between 66–67%. This cliff marks the boundary where structurally critical weights begin to be removed.

### Finding 2 — Dataset Complexity Determines Cliff Location and Shape

| Sparsity | MNIST Accuracy | FashionMNIST Accuracy |
|---|---|---|
| 0% (baseline) | 98.77% | 89.06% |
| 40% | 98.56% | 86.02% |
| 60% | 96.99% | 75.35% |
| 70% | 86.77% | 66.37% |
| 80% | 56.83% | 33.16% |

MNIST shows a late, sharp cliff at 66–67% sparsity. FashionMNIST does **not** show gradual-only decline — a real, sharp cliff exists at 79.6–85% sparsity, with single steps losing 8+ accuracy points in just 0.1% of sparsity. The cliff is dataset-dependent in both location and sharpness, not architecture-fixed.

> Note: Earlier versions of this README incorrectly characterised FashionMNIST as "gradual decline only, no sharp cliff." Corrected following finer-resolution experiments (notebook 04).

### Finding 3 — Layer Sensitivity is Dataset Dependent

Single-layer isolated pruning at 70% sparsity (one layer pruned at a time, all others untouched):

| Layer | MNIST (70% pruned) | FashionMNIST (70% pruned) |
|---|---|---|
| conv1 | 98.46% | 78.22% ← most sensitive |
| conv2 | 97.98% | 82.80% |
| fc1 | 98.68% ← most robust | 88.45% ← most robust |
| fc2 | 97.03% ← most sensitive | 87.25% |

Simple tasks → decision layers (fc2) most sensitive.  
Complex tasks → feature extraction layers (conv1) most sensitive.  
**Layer sensitivity is not a fixed property of the architecture — it is shaped by the dataset.**

Multi-checkpoint sweep (Finding 5) shows rankings are not stable across sparsity levels on FashionMNIST — two ranking flips occur as sparsity increases toward the cliff.

### Finding 4 — Sensitivity-Aware Smart Pruning Outperforms Uniform Pruning

| Method | Sparsity | MNIST | FashionMNIST |
|---|---|---|---|
| Uniform pruning | ~79% | 63.22% | 58.88% |
| Smart pruning (no retraining) | 79.3% | 92.39% | 75.06% |
| Smart pruning + retraining | 79.3% | **98.70%** | **89.11%** |

Smart pruning + layer-wise retraining achieves near-baseline accuracy at 79.3% sparsity — less than 1% accuracy loss on both datasets.

### Finding 5 — Dataset Complexity Governs Whether Sensitivity Gap Predicts the Cliff

Multi-checkpoint single-layer sensitivity sweep: 21 checkpoints on MNIST (60–80%), 26 checkpoints on FashionMNIST (0–85%), measuring the sensitivity gap (most-sensitive minus least-sensitive layer accuracy) at each sparsity level.

**MNIST:**
- Gap stays flat within baseline noise (~0.6–0.9 points) through 65% sparsity.
- Departs from baseline at 67% — the **same checkpoint** as the cliff itself.
- Sensitivity gap is a **concurrent signal**, not an early-warning one.
- Layer ranking (fc2 most sensitive, fc1 most robust) never flips across the entire 60–80% range.
- Isolated single-layer damage at 67% accounts for only ~1 accuracy point, while joint pruning at 67% causes ~7 points of damage — pointing to layer-interaction effects that single-layer testing cannot capture.

**FashionMNIST:**
- Gap stays flat through ~40% (baseline noise ≤1.93 points).
- Departs clearly from baseline at **~50–55% sparsity** — approximately **25 percentage points before** the confirmed cliff at 79.6%.
- Sensitivity gap is a **genuine early-warning signal** on this dataset.
- Two ranking flips confirmed: conv1 weakest (45–72%), conv2 takes over (73–80%), conv1 retakes (81–85%).
- Peak gap (~61 points at 85%) is roughly 4× MNIST's peak (~15.6 points at 80%).

**Cross-dataset conclusion:**  
Dataset complexity appears to govern whether the sensitivity gap provides early warning of the cliff (FashionMNIST — genuine early signal ~25pp ahead) or only concurrent confirmation (MNIST — signal arrives with the cliff, not before). This is an n=2 observation. A third dataset is needed to test generalisability.

---

## Connection to Literature

| Paper | Finding | How This Study Extends It |
|---|---|---|
| Han et al. 2015 | Magnitude pruning works | Found exact cliff threshold + non-uniform layer tolerance |
| Frankle & Carlin 2019 | Winning ticket subnetwork exists | Located where winning ticket concentrates by layer and dataset |
| Blalock et al. 2020 | Pruning results are fragmented, cliff not studied | Directly characterises cliff location and its relationship to layer sensitivity |
| Pesce, He & Caldarelli 2026 | Dataset-complexity-dependent cliff confirmed at whole-network level | This study extends to layer-level: sensitivity gap as early-warning signal |
| Hu, Gibson & Cano 2023 (ICE-Pick) | Layer sensitivity ordering stable across pruning steps (fixed dataset) | This study varies dataset, not architecture — the complementary untested axis |

---

## Model

**TinyNet** — custom CNN designed for TinyML scale experiments:
- 2 Conv layers (8 and 16 filters, 3×3)
- 2 FC layers (64 → 10)
- Total parameters: 52,138
- Reflects real TinyML deployment constraints

## Datasets
- **MNIST** — handwritten digit classification (10 classes, 60K train / 10K test)
- **FashionMNIST** — clothing classification (10 classes, 60K train / 10K test)

---

## Repository Structure
tinyml-pruning-study/
├── 01_baseline_training.ipynb               # TinyNet training on MNIST and FashionMNIST
├── 02_pruning_experiments.ipynb             # Uniform pruning, sensitivity analysis, smart pruning (MNIST)
├── 03_fashionmnist_experiments.ipynb        # Cross-dataset generalisation experiments
├── 04_layer_sensitivity_checkpoints.ipynb   # Multi-checkpoint sensitivity gap — MNIST + FashionMNIST
├── pruning_results.png                      # Accuracy vs sparsity curve
├── mnist_vs_fashion_pruning.png             # Cross-dataset comparison graph
├── Sensitivitygap_vs_sparsity(MNIST).png    # Sensitivity gap vs sparsity — MNIST, all 21 checkpoints
├── Sensitivitygap_vs_sparsity(fashionMNIST).png  # Sensitivity gap vs sparsity — FashionMNIST, full
├── tinynet_baseline.pth                     # Saved MNIST baseline model weights
├── tinynet_fashion_baseline.pth             # Saved FashionMNIST baseline model weights
└── README.md
---

## Foundational Papers

1. Han et al. 2015 — *Learning both Weights and Connections for Efficient Neural Networks* ✅
2. Frankle & Carlin 2019 — *The Lottery Ticket Hypothesis* (reading in progress)
3. Blalock et al. 2020 — *What is the State of Neural Network Pruning?* ✅
4. Hu, Gibson & Cano 2023 — *ICE-Pick: Iterative Cost-Efficient Pruning* ✅
5. Pesce, He & Caldarelli 2026 — *Phase Transitions in Neural Network Pruning* ✅

---

## LinkedIn Research Posts

- [Post 1 — Smart Pruning vs Uniform Pruning](https://www.linkedin.com/feed/update/urn:li:activity:7469001090680754176/)
- [Post 2 — Does Dataset Complexity Determine Compression Limits?](https://www.linkedin.com/feed/update/urn:li:activity:7469719195765628928/)

---

## Next Steps

- [ ] Verify conv1@71% anomaly on FashionMNIST (V-shaped spike needs standalone re-run)
- [ ] Run experiment on a third dataset to test early-warning pattern generalisability
- [ ] LinkedIn post on FashionMNIST cliff correction and Finding 5
- [ ] Schedule chat with Prof. José Cano (University of Glasgow)
- [ ] Follow up with Prof. Eiman Kanjo with new results
- [ ] Explore structured pruning as thesis/PhD extension direction

---

*This is independent research conducted as part of preparation for PhD applications in TinyML and efficient deep learning.*