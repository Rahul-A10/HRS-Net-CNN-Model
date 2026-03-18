# HRSNet: A Fusion of HRNet and ResNet

> A hybrid image classification model combining High-Resolution Network (HRNet) multi-scale feature representation with ResNet-style residual connections at the stage level.

**Authors:** Rahul Autade  · Raj Surya Rajendran Kathirvel 
**Institution:** University of Minnesota

---

## Overview

HRSNet modifies the standard HRNet architecture by wrapping each stage (2, 3, and 4) as a residual unit — adding the stage's input directly to its output. This allows gradients to flow more freely through the deep multi-branch network, improving convergence and classification accuracy on CIFAR-100.

## Architecture

- **Stem:** 2 conv layers, reduces spatial resolution by 1/4
- **Stage 1:** 4 Bottleneck blocks, 1 branch
- **Stages 2–4:** 4 Modularized blocks each, with 2/3/4 parallel branches respectively
- **Fusion (HRSNet):** Each stage wrapped with a residual skip connection (input + output)
- **Head:** Global average pooling → 100-class classifier

Each branch `k` maintains `(2^(k-1)) * C` feature maps. One Modularized Block = 4 Residual Units.

## Dataset

**CIFAR-100** — chosen over ImageNet for computational feasibility.

| Property | Value |
|---|---|
| Images | 60,000 (500 train / 100 test per class) |
| Classes | 100 (fine) / 20 (coarse) |
| Input size | 32×32 → resized to 224×224 |
| Dataset size | ~170 MB |

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| LR Patience | 2 |
| Epochs | ~10 |
| C (channel width) | 2, 18, 30 |

## Results

HRSNet consistently outperforms baseline HRNet across all channel widths. At C=30, HRSNet reaches ~55% top-1 accuracy vs ~54% for HRNet.

**Best classes:** wardrobe (85%), sunflower (84%), plain (81%)  
**Hardest classes:** girl (2%), otter (12%), lizard (13%)

## Future Work

- Hyperparameter search beyond C ∈ {2, 18, 30}
- Benchmarking on ImageNet, CIFAR-10, and other datasets
- Extension to semantic segmentation and pose estimation tasks
