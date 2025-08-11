import torch
import os

def save_checkpoint(model, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, filename))
    print(f"✅ Model saved to {os.path.join(out_dir, filename)}")
