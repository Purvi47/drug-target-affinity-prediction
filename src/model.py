import torch
import torch.nn as nn

class DeepDTA(nn.Module):
    def __init__(self, smiles_vocab_size, seq_vocab_size, smiles_len, seq_len):
        super().__init__()
        emb_dim = 128

        self.smiles_emb = nn.Embedding(smiles_vocab_size+1, emb_dim, padding_idx=0)
        self.seq_emb = nn.Embedding(seq_vocab_size+1, emb_dim, padding_idx=0)

        self.smiles_conv = nn.Sequential(
            nn.Conv1d(emb_dim, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.seq_conv = nn.Sequential(
            nn.Conv1d(emb_dim, 32, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(smiles_len, 4) + self._get_conv_output(seq_len, 8), 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1)
        )

    def _get_conv_output(self, length, kernel):
        return 32 * ((length - kernel + 1) // 2)

    def forward(self, smiles, seq):
        s = self.smiles_emb(smiles).permute(0, 2, 1)
        s = self.smiles_conv(s).view(s.size(0), -1)

        p = self.seq_emb(seq).permute(0, 2, 1)
        p = self.seq_conv(p).view(p.size(0), -1)

        x = torch.cat([s, p], dim=1)
        return self.fc(x)
