# Drug-Target Binding Affinity Prediction (drug-target-dta)

This repository contains a deep learning project to predict drug-target binding affinity using the KIBA dataset. The goal is to estimate the binding strength between a drug molecule (represented by SMILES) and a target protein (represented by its amino acid sequence).

---

## Project Structure
```drug-target-dta/
├─ data/
│ └─ kiba.csv # Dataset containing SMILES, protein sequences, and affinity values
├─ src/
│ ├─ init.py
│ ├─ config.py # Configuration parameters
│ ├─ dataset.py # Dataset processing and tokenization
│ ├─ model.py # Deep learning model definition
│ ├─ train.py # Training script
│ ├─ evaluate.py # Model evaluation script
│ └─ utils.py # Utility functions
├─ outputs/
│ ├─ models/
│ │ └─ best_model.pt # Trained model checkpoint
│ └─ logs/ # Training logs and metrics
├─ requirements.txt # Python dependencies
├─ README.md # Project documentation
└─ run.sh # Shell script to run training/evaluation
```
