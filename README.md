# Heterogeneous Federated Green Learning (HFGL) for Cell-Free MIMO Beamforming

This repository contains the Python implementation of the Heterogeneous Federated Green Learning (HFGL) framework for cell-free Massive MIMO beamforming, as well as the synthetic dataset generator designed for simulating channel state information (CSI) and optimizing sum-rate performance under resource-constrained and heterogeneous edge environments.

---

## Overview

Traditional deep learning-based beamforming requires high computational resources and centralized channel data, introducing high latency, severe fronthaul overhead, and privacy risks.

This project implements an interpretable, low-complexity, and communication-efficient Federated Green Learning (FGL) framework. Key capabilities include:

- Subspace Approximation with Adjusted Bias (Saab) / PCA: Multi-stage feature extraction for spatially distributed pilot observations.
- Heterogeneous & Distributed PCA (HPCA / DPCA): Enables local subspace approximation across distributed Access Points (APs) with non-i.i.d. data and varying capacity ratios.
- Relevant Feature Test (RFT): Supervised feature selection method to select the most informative features with minimal computational overhead.
- Heterogeneous Federated Learning: Aggregates models with varying feature input sizes across client APs without requiring raw CSI sharing.

*Note: This repository provides source code only and does not include pre-generated dataset files. Users must run the dataset generator script to create their own dataset and specify filenames according to their specific requirements.*

---

## Repository Structure
### 1. cell_free_data_generator20260911.py
- Generates 2D/3D geometry-based channel state information (CSI) with path loss (3GPP Urban Micro/Macro model) and Rician fading.
- Computes MMSE precoding vectors and performs Water-filling power allocation.
- Simulates pilot signal reception under imperfect CSI conditions.
- Calculates sum-rate benchmarks and exports structured datasets (.npz).

> **Note on Generator Parameters**: The default parameters in `cell_free_data_generator20260911.py` are intended solely for quick testing, sanity checks, and code demonstration. They **do not** correspond to the full-scale simulation parameters used in the paper. If you wish to reproduce the exact experimental results from the paper, please modify the generation parameters (e.g., number of APs/UEs, antennas, coverage area, sample sizes, and SNR) accordingly.

### 2. HFGL20260911.py
- Class AP: Core object handling local datasets, modular green learning steps, and federated aggregation.
- Module 1 (HPCA / DPCA / Saab): Spatial-temporal subspace feature extraction using standard, federated (FedAvg), or heterogeneous block power methods.
- Module 2 (RFT Selection): Computes RFT loss to rank and select top-K representative features.
- Module 3 (Heterogeneous FedAvg / Regression): Fits lightweight prediction models (MLP, XGBoost, or Heterogeneous Neural Networks) via distributed federated rounds to predict beamforming vectors.
- Sum-Rate Reconstruction: Validates predicted precoding matrices against actual channels to compute sum-rate loss and performance.
