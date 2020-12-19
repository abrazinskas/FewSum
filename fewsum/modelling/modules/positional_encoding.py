import torch
import math


class PositionalEncoding(torch.nn.Module):
    """
    Returns embeddings of words which also contains the position (time)
    component.
    """

    def __init__(self, emb_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # initializing the lookup table
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (
                    -math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, word_embs, offset=0):
        """
        Args:
            word_embs: [batch_size, seq_len, embd_dim]
            offset: self-explanatory.

        Returns:
            out: [batch_size, seq_len, embd_dim]
        """
        max_len = word_embs.size(1)
        out = word_embs + self.pe[offset:max_len + offset].unsqueeze(0)
        return out
