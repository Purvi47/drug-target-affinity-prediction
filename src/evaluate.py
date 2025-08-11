import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from .dataset import KibaDataset
from .model import DeepDTA
from .config import *

def main():
    # Load dataset
    dataset = KibaDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=64)

    # Load trained model
    model = DeepDTA(len(SMILES_VOCAB), len(SEQ_VOCAB), MAX_SMILES_LEN, MAX_SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt")))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for smiles, seq, label in loader:
            smiles, seq, label = smiles.to(DEVICE), seq.to(DEVICE), label.to(DEVICE)
            outputs = model(smiles, seq).squeeze().cpu().numpy()
            labels = label.cpu().numpy()

            preds.extend(outputs)
            targets.extend(labels)

    preds = np.array(preds)
    targets = np.array(targets)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    pearson_corr, _ = pearsonr(targets, preds)

    print(f"Sample predictions: {preds[:5]}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Scatter plot: predicted vs true
    plt.subplot(1, 3, 1)
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel("True Affinity")
    plt.ylabel("Predicted Affinity")
    plt.title("Predicted vs True Affinity")

    # Residual plot
    plt.subplot(1, 3, 2)
    residuals = targets - preds
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Affinity")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residual Plot")

    # Error distribution histogram
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual Error")
    plt.ylabel("Frequency")
    plt.title("Residual Error Distribution")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
