import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from .dataset import KibaDataset
from .model import DeepDTA
from .config import *
from .utils import save_checkpoint

def main():
    print("[INFO] Loading dataset...")
    dataset = KibaDataset(DATA_PATH)
    print(f"[INFO] Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = DeepDTA(len(SMILES_VOCAB), len(SEQ_VOCAB), MAX_SMILES_LEN, MAX_SEQ_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\n[INFO] Starting epoch {epoch+1}/{EPOCHS}...")
        model.train()
        train_loss = 0.0
        for batch_idx, (smiles, seq, y) in enumerate(train_loader):
            smiles, seq, y = smiles.to(device), seq.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(smiles, seq).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * smiles.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for smiles, seq, y in val_loader:
                smiles, seq, y = smiles.to(device), seq.to(device), y.to(device)
                preds = model(smiles, seq).squeeze()
                loss = criterion(preds, y)
                val_loss += loss.item() * smiles.size(0)

        train_loss /= train_size
        val_loss /= val_size
        print(f"[INFO] Epoch {epoch+1} - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, MODEL_DIR, "best_model.pt")
            print(f"[INFO] Best model saved with val_loss {val_loss:.4f}")

if __name__ == "__main__":
    main()
