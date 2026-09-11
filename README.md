# Cell-Free-MIMO-Beamforming-With-Federated-Green-Learning
This is the simulation code of the paper.
# Federated Green Learning for Cell-Free Massive MIMO

This repository provides the implementation of a Federated Green Learning (FGL) framework for downlink beamforming in UE-centric cell-free massive MIMO systems.

The framework combines lightweight Green Learning (GL) modules with Federated Learning (FL) to enable distributed model training while reducing computational and communication costs.

## Overview

The FGL framework consists of three main modules:

- **Module 1:** PCA-based feature transformation using distributed PCA (DPCA)
- **Module 2:** RFT-based feature selection
- **Module 3:** Lightweight MLP-based beamforming prediction

The framework also supports heterogeneous clients with different computational capabilities. In the heterogeneous setting, resource-constrained clients can operate with a reduced feature dimension while the global model maintains the full feature dimension.

The repository also includes centralized and federated neural-network baselines for performance comparison.

## Repository Contents

```text
.
├── cell_free_data_generator20260911.py
├── HFGL20260911.py
└── README.md

Cell-Free MIMO Simulation
          │
          ▼
     Received Pilots
          │
          ▼
       Module 1
   Distributed PCA
          │
          ▼
       Module 2
    RFT Feature Selection
          │
          ▼
       Module 3
     MLP Prediction
          │
          ▼
   Beamforming Vectors
          │
          ▼
      Sum Rate
 Client 1 ──┐
 Client 2 ──┤
 Client 3 ──┼──► Federated Aggregation ──► Global Model
 Client 4 ──┤
 Client 5 ──┘
