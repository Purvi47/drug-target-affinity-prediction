import os
import torch
# Paths
DATA_PATH = os.path.join("data", "kiba.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model hyperparameters
MAX_SMILES_LEN = 100
MAX_SEQ_LEN = 1000
SMILES_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#=()[]+-/\\."
SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
