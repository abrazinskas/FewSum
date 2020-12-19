from torch.nn import Module, Embedding, Dropout
from .positional_encoding import PositionalEncoding
import math
import torch as T


class TransformerEmbeddings(Module):
    """Performs projection to the embedding space and selection;

    Embedding of sequences (embed method):
        -   The actual embedding of sequences.
        -   Rescaling by multiplying by emb_dim**0.5.
        -   Adding positional embeddings.
        -   Dropout on the level of vectors.

    Projection/forward pass:
        -   Rescaling of the embd matrix by multiplying by emb_dim**0.5.
        -   Dropout on the level of vectors.
        -   Dot product with the embd matrix.
    """
    def __init__(self, vocab_size, word_embd_dim, len_embd_dim=None,
                 dropout=0.0, max_len=5000):
        super(TransformerEmbeddings, self).__init__()
        self.embs = Embedding(vocab_size, word_embd_dim)
        self.dropout = Dropout(dropout)
        self.len_embd_dim = len_embd_dim
        self.max_len = max_len
        if len_embd_dim is not None:
            self.pos_enc = Embedding(max_len, len_embd_dim)
        else:
            self.pos_enc = PositionalEncoding(word_embd_dim, max_len)
        self.word_embd_dim = word_embd_dim

    def _rescale(self, embd):
        return embd * math.sqrt(self.word_embd_dim)

    def _pos_enc(self, embd, offset=0):
        if offset >= self.max_len:
            raise ValueError("Offset (%d) needs to be smaller than the maximum "
                             "allowed length (%d)." % (offset, self.max_len))
        if self.len_embd_dim is not None:
            batch_size = embd.size(0)
            max_len = embd.size(1)
            dummy = T.arange(offset, max_len+offset, dtype=T.int64,
                             device=embd.device)
            len_embd = self.pos_enc(dummy).unsqueeze(0)\
                .repeat(batch_size, 1, 1)
            embd = T.cat((embd, len_embd), dim=-1)
            return embd
        else:
            return self.pos_enc(embd, offset)

    def _embed(self, x, pos_offset=0, dropout=True):
        """
        Args:
            x (LongTensor): [batch_size, seq_len]
            pos_offset (int): offset for positional embeddings.
        Returns:
            out (FloatTensor): [batch_size, seq_len, emb_dim]
        """
        embd = self.embs(x)
        embd = self._rescale(embd)
        embd = self._pos_enc(embd, offset=pos_offset)
        if dropout:
            out = self.dropout(embd)
        else:
            out = embd
        return out

    def _project(self, x, dropout=True):
        """
        Args:
            x (FloatTensor): [batch_size, seq_len, emb_dim]
        Returns:
            out (FloatTensor): [batch_size, seq_len, vocab_size]
        """
        sel = self._rescale(self.embs.weight)
        if dropout:
            sel = self.dropout(sel)
        out = T.matmul(x, sel.t())
        return out

    def forward(self, x, pos_offset=0, dropout=True):
        if x.dtype == T.float32:
            return self._project(x, dropout=dropout)
        elif x.dtype in [T.int32, T.int64]:
            return self._embed(x, dropout=dropout, pos_offset=pos_offset)
        else:
            raise NotImplementedError
