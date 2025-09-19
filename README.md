# DeepSeq3

DeepSeq3 is a tool for sequencial circuit analysis.  
This repository provides a quick start workflow: simply prepare your data and run the provided script.

---

## Installation

This project requires **Python 3.8+** and CUDA 11.8 for GPU acceleration (if using PyTorch with GPU).

### 1. Clone the repository

```bash
git clone https://github.com/zhoujy22/DeepSeq3.git
cd <your-repo-directory>
```

### 2. Install dependencies

All required Python packages are listed in requirements.txt. Install them using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Preparing datasets for specific experiments

For comgraph experiments
```bash
python src/prepare_dataset_comgraph.py
```
For SNG experiments
```bash
python src/prepare_dataset_sng.py
```

## Quick Start

After installing dependencies and preparing your data in `raw_data/`, you can quickly run the experiments:

The main startup script `ds3.sh` will run the stage_1 default experiment:

```bash
bash run/ds3.sh
```
This will automatically load datasets from raw_data/ and start training the stage_1 default model.

