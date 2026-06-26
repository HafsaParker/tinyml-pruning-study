# TinyML Pruning Study: Layer Sensitivity Analysis for Efficient Model Compression

**Author:** Hafsa Parker  
**Type:** Independent Research  
**Status:** Experiments Complete — Write-up in Progress

---

## Overview

This project presents an empirical study on neural network pruning for TinyML scale networks. Starting from Han et al. 2015 — the foundational magnitude pruning paper — this study goes beyond reproduction to investigate **where pruning tolerance breaks down, why it breaks down there, and how sensitivity-aware pruning can recover accuracy.**

The central question:

> *Why hasn't pruning research converged — and where are the remaining gaps?*

---

## Key Findings

### Finding 1 — Accuracy Cliff at 66-67% Sparsity
Uniform magnitude pruning maintains accuracy until a critical threshold. For MNIST, accuracy stays above 95% up to 65% sparsity, then collapses sharply between 66-67%. This cliff marks the boundary where critical (lottery ticket) weights begin to be removed.

### Finding 2 — Dataset Complexity Determines Pruning Tolerance
| Sparsity | MNIST Accuracy | FashionMNIST Accuracy |
|---|---|---|
| 0% (baseline) | 99% | 89.70% |
| 40% | 98.56% | 87.20% |
| 60% | 96.99% | 74.86% |
| 70% | 86.77% | 58.88% |
| 80% | 56.83% | 29.38% |

Simple datasets tolerate aggressive pruning. Complex datasets degrade from 40% sparsity onwards — no sharp cliff, just steady decay.

### Finding 3 — Layer Sensitivity is Dataset Dependent
| Layer | MNIST Accuracy (70% pruned) | FashionMNIST Accuracy (70% pruned) |
|---|---|---|
| conv1 | 98.46% | 70.53% ← most sensitive |
| conv2 | 97.98% | 84.77% |
| fc1 | 98.68% ← most robust | 89.15% ← most robust |
| fc2 | 97.03% ← most sensitive | 82.32% |

For simple tasks — decision layers are most sensitive.  
For complex tasks — feature extraction layers are most sensitive.  
**Layer sensitivity is not fixed — it is determined by dataset complexity.**

### Finding 4 — Sensitivity-Aware Smart Pruning Outperforms Uniform Pruning
| Method | Sparsity | MNIST | FashionMNIST |
|---|---|---|---|
| Uniform pruning | ~79% | 63.22% | 58.88% |
| Smart pruning (no retraining) | 79.3% | 92.39% | 75.06% |
| Smart pruning + retraining | 79.3% | **98.70%** | **89.11%** |

Smart pruning + layer-wise retraining achieves near-baseline accuracy at 79.3% sparsity on both datasets — less than 1% accuracy loss.

---

## Connection to Literature

| Paper | Finding | How This Study Extends It |
|---|---|---|
| Han et al. 2015 | Magnitude pruning works | Found exact cliff threshold + non-uniform layer tolerance |
| Frankle & Carlin 2019 | Winning ticket subnetwork exists | Located where winning ticket concentrates by layer and dataset |

---

## Model

**TinyNet** — custom CNN designed for TinyML scale experiments:
- 2 Conv layers (8 and 16 filters, 3x3)
- 2 FC layers (64 → 10)
- Total parameters: 52,138
- Reflects real TinyML deployment constraints

## Datasets
- **MNIST** — handwritten digit classification (10 classes, 60K train / 10K test)
- **FashionMNIST** — clothing classification (10 classes, 60K train / 10K test)

---

## Repository Structure

```
tinyml-pruning-study/
├── 01_baseline_training.ipynb        # TinyNet training on MNIST
├── 02_pruning_experiments.ipynb      # Uniform pruning, sensitivity analysis, smart pruning
├── 03_fashionmnist_experiments.ipynb # Cross-dataset generalisation experiments
├── pruning_results.png               # Accuracy vs sparsity curve
├── mnist_vs_fashion_pruning.png      # Cross-dataset comparison graph
├── tinynet_baseline.pth              # Saved MNIST baseline model
├── tinynet_fashion_baseline.pth      # Saved FashionMNIST baseline model
├── 04_layer_sensitivity_checkpoints.ipynb  # Multi-checkpoint sensitivity gap experiment
├── layer_sensitivity_gap_mnist_full.png    # Gap vs sparsity, all 21 checkpoints
├── layer_sensitivity_raw_mnist.png         # Per-layer accuracy, all 21 checkpoints
└── README.md
```

---

## Foundational Papers

1. Han et al. 2015 — *Learning both Weights and Connections for Efficient Neural Networks* ✅
2. Frankle & Carlin 2019 — *The Lottery Ticket Hypothesis* (reading in progress)
3. Blalock et al. 2020 — *What is the State of Neural Network Pruning?* (upcoming)
4. José Cano et al. 2023 — *ICE-Pick: Iterative Cost-Efficient Pruning* (upcoming)

---

## LinkedIn Research Posts

- [Post 1 — Smart Pruning vs Uniform Pruning](https://www.linkedin.com/feed/update/urn:li:activity:7469001090680754176/)
- [Post 2 — Does Dataset Complexity Determine Compression Limits?](https://www.linkedin.com/feed/update/urn:li:activity:7469719195765628928/)

---

## Next Steps

- [ ] Read Frankle 2019 and Blalock 2020
- [ ] Write 2-page research summary
- [ ] Explore structured pruning
- [ ] Thesis registration with supervisor

---
### Finding 5 — Layer Sensitivity Gap Tracks the Cliff, Does Not Predict It Early (MNIST)
Single-layer isolated pruning across 21 checkpoints (60-80% sparsity) shows the sensitivity
gap (most-sensitive minus least-sensitive layer accuracy) stays flat near baseline noise
(~0.6-0.9 points) through 65% sparsity, then departs from that baseline starting at 67% —
the same checkpoint where the uniform-pruning cliff itself begins, not earlier. The gap
continues widening well past the cliff, reaching >15 points by 80% sparsity.

fc2 (MNIST's most sensitive layer) drives nearly the entire gap throughout. Layer ranking
never flips across the full 60-80% range.

Isolated single-layer damage cannot account for the full magnitude of the joint cliff:
at 67% sparsity, pruning fc2 alone costs ~1 accuracy point, while pruning all four layers
together costs ~7 points — pointing to layer-interaction effects under joint pruning that
single-layer sensitivity testing cannot capture by design.

**Conclusion:** sensitivity and the cliff are entangled (consistent with Kanjo's framing),
but the gap functions as a concurrent signal here, not an early-warning one. Cross-dataset
test (FashionMNIST) pending.

*This is independent research conducted as part of preparation for PhD applications in TinyML and efficient deep learning.*
