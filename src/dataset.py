import torch
from torch.utils.data import Dataset
import pandas as pd
from .config import MAX_SMILES_LEN, MAX_SEQ_LEN, SMILES_VOCAB, SEQ_VOCAB

# Build vocab dicts once
SMILES_CHAR2IDX = {ch: i + 1 for i, ch in enumerate(SMILES_VOCAB)}
SEQ_CHAR2IDX = {ch: i + 1 for i, ch in enumerate(SEQ_VOCAB)}

def tokenize(text, char2idx, max_len):
    token_ids = [char2idx.get(ch, 0) for ch in text[:max_len]]
    token_ids += [0] * (max_len - len(token_ids))
    return token_ids

class KibaDataset(Dataset):
    def __init__(self, csv_path):
        # Load CSV
        self.data = pd.read_csv(csv_path)

        # Print number of missing values before dropping
        print("Missing values before dropping:")
        print(self.data[['smiles', 'sequence', 'affinity']].isnull().sum())

        # Drop rows with missing values in important columns
        self.data = self.data.dropna(subset=['smiles', 'sequence', 'affinity'])

        # Normalize affinity scores to [0,1]
        min_aff = self.data['affinity'].min()
        max_aff = self.data['affinity'].max()
        self.data['affinity'] = (self.data['affinity'] - min_aff) / (max_aff - min_aff)

        print(f"Original dataset size: {len(self.data) + self.data.isnull().sum().max()}")
        print(f"Dataset size after dropping missing values: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = tokenize(row["smiles"], SMILES_CHAR2IDX, MAX_SMILES_LEN)
        seq = tokenize(row["sequence"], SEQ_CHAR2IDX, MAX_SEQ_LEN)
        affinity = torch.tensor(row["affinity"], dtype=torch.float)
        return torch.tensor(smiles, dtype=torch.long), torch.tensor(seq, dtype=torch.long), affinity

if __name__ == "__main__":
    dataset = KibaDataset("data/kiba.csv")
    sample_smiles, sample_seq, sample_affinity = dataset[0]
    print("Sample SMILES tokens:", sample_smiles)
    print("Sample Sequence tokens:", sample_seq)
    print("Sample affinity:", sample_affinity.item())
