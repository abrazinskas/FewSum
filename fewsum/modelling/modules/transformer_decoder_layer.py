import torch as T
import copy
from torch.nn import functional as F
from torch.nn import Module, Sequential
from torch.nn.modules.activation import MultiheadAttention, ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class TransformerDecoderLayer(Module):
    """
    Modified version of the default PyTorch transformer decoder.
    Altered the output of the forward method and the way the multi-head attention
    works.
    """

    def __init__(self, model_dim, nheads, ff_dim=2048, dropout=0.1,
                 mem_att=True, mem_att_dropout=0.1, mem_dim=None):
        """
        Args:
            mem_att: whether to add a memory submodule.
            mem_dim: dimensionality of the memory to attend.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(model_dim, nheads, dropout=dropout)
        if mem_att:
            mem_dim = mem_dim if mem_dim is not None else model_dim
            self.mem_attn = MultiheadAttention(model_dim, nheads,
                                               dropout=mem_att_dropout,
                                               vdim=mem_dim, kdim=mem_dim)
        else:
            self.mem_attn = None

        # Implementation of Feedforward model
        self.linear1 = Linear(model_dim, ff_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(ff_dim, model_dim)

        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)
        self.norm3 = LayerNorm(model_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, mem=None, tgt_mask=None, mem_mask=None,
                tgt_key_padding_mask=None, mem_key_padding_mask=None,
                state=None):
        """
        Args:
            tgt (FloatTensor): the sequence to the decoder layer (required).
                [tgt_seq_len, batch_size, model_dim]
            mem (FloatTensor): the sequence from the last layer of the encoder
                (optional).
                [mem_seq_len, batch_size, model_dim]
            tgt_mask (FloatTensor): the additive mask for the tgt sequence with
                0. for unmasked and -inf for masked positions (optional).
                [tgt_seq_len, tgt_seq_len]
            mem_mask (FloatTensor): the additive mask for the memory sequence
                with 0. for unmasked and -inf for masked positions (optional).
                [tgt_seq_len, tgt_seq_len]
            tgt_key_padding_mask (ByteTensor): the mask for the tgt keys per
                batch (optional).
                [batch_size, tgt_seq_len]
            mem_key_padding_mask (ByTensor): the mask for the memory keys per
                batch (optional).
                [batch_size, mem_seq_len]
            state (FloatTensor): previously computed representations of the tgt.
                [state_seq_len, batch_size, model_dim]
            props (FloatTensor):
                [tgt_seq_len, batch_size, prop_dim]
        Returns:
            tgt (FloatTensor): computed representation of the tgt.
                [tgt_seq_len, batch_size, model_dim
            mem_att_wts (FloatTensor):
                [tgt_seq_len, batch_size]
        """
        if state is None:
            tgt2, self_att_wts = self.self_attn(tgt, tgt, tgt,
                                                attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
        else:
            tgt2, self_att_wts = self.self_attn(tgt, state, state,
                                                attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # below is the case when the memory is present, i.e., encoding of
        # input is available
        if mem is not None and self.mem_attn is not None:
            tgt2, mem_att_wts = self.mem_attn(tgt, mem, mem, attn_mask=mem_mask,
                                              key_padding_mask=mem_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        else:
            mem_att_wts = None

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, mem_att_wts
