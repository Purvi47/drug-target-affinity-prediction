import pandas as pd
import os

# Input/output paths
RAW_PATH = os.path.join("data", "kiba.txt")
OUT_PATH = os.path.join("data", "kiba.csv")

rows = []
with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # Extract needed parts
        smiles = parts[2]           # SMILES string
        sequence = parts[3]         # FASTA sequence
        affinity = parts[4]         # KIBA score

        rows.append({
            "smiles": smiles,
            "sequence": sequence,
            "affinity": float(affinity)
        })

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"✅ Saved CSV to {OUT_PATH} with {len(df)} rows")
