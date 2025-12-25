# WDP-Mamba: Wavelet-Augmented Dual-Branch Position-Embedding Mamba Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Task-Hyperspectral_Change_Detection-green.svg)]()

This repository contains a complete PyTorch implementation of **WDP-Mamba**, based on the paper:

> **A Wavelet-Augmented Dual-Branch Position-Embedding Mamba Network for Hyperspectral Image Change Detection** > *Chen Ding, Xiaofeng Hao, Sirui Zheng, Yizhou Dong, Wenqiang Hua, Wei Wei, Lei Zhang, and Yanning Zhang* > IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2025.

## 📖 Introduction

Hyperspectral Image Change Detection (HSI-CD) often struggles with the trade-off between extracting local fine-grained details and modeling global semantic dependencies. 

**WDP-Mamba** addresses this by combining:
1.  **Global Branch (Mamba):** Uses a State Space Model (SSM) with a **4-Directional Selective Scan** to capture long-range dependencies with linear complexity.
2.  **Local Branch (Wavelet):** Uses a **Multi-Level Frequency-Aware (MLFA)** module based on Discrete Wavelet Transforms (DWT) to capture frequency-specific local details.
3.  **Adaptive Position Embeddings (APE):** Preserves spatial structure often lost in standard Mamba flattening operations.

## ✨ Key Features

* **Pure PyTorch Implementation:** The State Space Model (SSM) kernel is implemented in native PyTorch, removing the need for difficult-to-install custom CUDA kernels (e.g., `mamba-ssm`) for experimentation.
* **Corrected Wavelet Module:** Includes a robust `WTConvLayer` that handles dimension matching for concatenation correctly.
* **Full Metrics:** Automatically calculates OA, Kappa, F1, IoU, Precision, and Recall.
* **Visualization:** Automatically saves Ground Truth and Predicted Change Maps as images.

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* PyTorch 1.12+

### Install Dependencies
```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib
