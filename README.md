# Harnessing Graph Learning for Surfactant Chemistry: PharmHGT, GCN, and GAT in LogCMC Prediction

Welcome to this repository! It encompasses code, test dataset and resources for graph neural network-based machine learning approaches to predict the logarithm of Critical Micelle Concentration (LogCMC) in surfactant chemistry.


## Directory Structure

### Models
- **GAT Models:**
  - `GAT_early_stop_skipCV_Split70_20_10_data1_best_opt2`: GAT model with early stopping and cross-validation (Dataset 1)
  - `GAT_early_stop_skipCV_Split70_20_10_data2_best_opt2`: GAT model with early stopping and cross-validation (Dataset 2)

- **GCN Models:**
  - `GCN_early_stop_skipCV_split70_20_10_data1_best_opt2`: GCN model with optimized hyperparameters (Dataset 1)
  - `GCN_early_stop_skipCV_split70_20_10_data2_best_opt2`: GCN model with optimized hyperparameters (Dataset 2)

- **PharmHGT Models:**
  - `surfactant_model_data1_newSplit70_20_10_plot_opt_R4`: PharmHGT model for surfactant prediction (Dataset 1)
  - `surfactant_model_data2_Split70_20_10_plot_best_R2`: PharmHGT model for surfactant prediction (Dataset 2)

### Source Code
- **GCN-GAT:**
  - `GNN_functions.py`: Core graph neural network functions and utilities
  - `GNN_workflow.py`: Complete workflow for GNN training and evaluation
  - `NNgraph.py`: Neural network graph implementations

- **PharmHGT:**
  - `data.py`: Data preprocessing and handling functions
  - `model.py`: PharmHGT model architecture implementation
  - `train.py`: Training scripts and procedures

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Graph-transformers-GCN-GAT/GCN-GAT-PharmaHGT.git
cd GCN-GAT-PharmaHGT
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```


### Model Evaluation

Pre-trained models are available in the `models/` directory. Load and evaluate using the respective workflow scripts.

## Features

- **Graph Neural Networks**: Implementation of GAT and GCN architectures for molecular property prediction
- **PharmHGT**: Novel pharmacophore-aware heterogeneous graph transformer for surfactant analysis
- **Cross-Validation**: Robust model evaluation with stratified cross-validation
- **Early Stopping**: Prevents overfitting during training

## Requirements

See `requirements.txt` for detailed package dependencies. Key requirements include:
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- Scikit-learn

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{marchan2024harnessing,
  author = {G. Theis Marchan and T. Olayiwola and K. Territo and T. O. Balogun and R. Kumar and J. A. Romagnoli},
  title = {Harnessing Graph Learning for Surfactant Chemistry: PharmHGT, GCN, and GAT in LogCMC Prediction},
  journal = {Submitted to Digital Discovery},
  year = {2024},
  note = {Manuscript in preparation}
}
```
