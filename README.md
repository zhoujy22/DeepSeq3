# DeepSeq3

DeepSeq3 is a tool for sequencial circuit analysis.  
This repository provides a quick start workflow: simply prepare your data and run the provided script.

---

## Project Structure
├── raw_data/ # Folder for raw input data
│ └── example/ # Example dataset
│ └── example.bench # Example benchmark file
│
├── run/ # Shell scripts for running experiments
│ └── ds3.sh # Main startup script
│
├── src/ # Source code
│ ├── datasets/ # Dataset processing scripts
│ ├── models/ # Model definitions
│ ├── trains/ # Training logic
│ ├── utils/ # Utility functions and helpers
│ │
│ ├── config.py # Configuration file
│ ├── finetune_pe_main.py # Script for fine-tuning
│ ├── fsm_simulator.* # FSM simulator (compiled extension)
│ ├── main.py # Main entry point
│ ├── main_2.py # Alternative main entry point
│ ├── prepare_dataset_ablation.py # Dataset preparation for ablation studies
│ ├── prepare_dataset_comgraph.py # Dataset preparation for comgraph
│ └── prepare_dataset_sng.py # Dataset preparation for SNG
