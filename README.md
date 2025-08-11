# Drug-Target Binding Affinity Prediction (drug-target-dta)

This repository contains a deep learning project to predict drug-target binding affinity using the KIBA dataset. The goal is to estimate the binding strength between a drug molecule (represented by SMILES) and a target protein (represented by its amino acid sequence).

## Progress So Far

- Collected and saved the KIBA dataset (`kiba.csv`) in the `data/` folder.
- Implemented dataset processing and tokenization in `src/dataset.py` to prepare input data (SMILES and protein sequences) for the model.
- Defined the deep learning model architecture in `src/model.py` for predicting drug-target binding affinity.
- Created training logic in `src/train.py` to train the model using the processed dataset.
- Developed evaluation logic in `src/evaluate.py` to assess model performance on test data.
- Added configuration settings and hyperparameters in `src/config.py`.
- Wrote utility functions in `src/utils.py` to support data handling and model training.
- Organized project files into separate directories (`data/`, `src/`, `outputs/`) for clarity and maintainability.
- Configured output saving: best model checkpoints are saved in `outputs/models/`, and training logs in `outputs/logs/`.
- Created a shell script (`run.sh`) to simplify running training and evaluation processes.
- Listed all required Python packages in `requirements.txt` for easy environment setup.

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
