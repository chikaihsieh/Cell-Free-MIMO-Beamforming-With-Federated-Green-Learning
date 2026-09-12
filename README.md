# Cell-Free MIMO Beamforming With Federated Green Learning

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

This script implements the FGL framework and the baseline learning models.

After entering the `main` function, the script loads the datasets corresponding to **M = 1, 2, 3, 4, and 5 UEs**. Each configuration contains **20,000 samples**, resulting in a total of **100,000 samples**. These samples are then organized and combined into a unified dataset, with each UE configuration accounting for **20% of the total data**, before the subsequent training, validation, and evaluation procedures.

After model training, the script evaluates the trained model on the corresponding test dataset and calculates the **testing sum rate** based on the predicted beamforming vectors.

The main components include:

- Module 1: Saab-based feature transformation
- Module 2: RFT feature selection
- Module 3: MLP beamforming prediction
- Federated averaging (FedAvg)
- Heterogeneous federated averaging
- Centralized CNN baseline
- Testing sum-rate calculation

### Training and Validation Loss

If training and validation loss curves are needed, the training history can be obtained directly from the TensorFlow/Keras `History` object returned by the training process.

For example:

```python
history = model.fit(...)

### Using the CNN Baseline

The CNN baseline is included in `HFGL20260911.py` but is not enabled by default.

To use the CNN baseline, manually uncomment `NN_operator` and its corresponding prediction function in the script. This allows the CNN model to be trained and evaluated using the same dataset and evaluation procedure.
