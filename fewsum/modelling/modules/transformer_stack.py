import torch as T
from torch.nn.modules.transformer import _get_clones, LayerNorm
from fewsum.modelling.modules import TransformerDecoderLayer
from torch.nn import Module, Sequential, Linear


MEM_ATT_WTS = 'mem_att_wts'


class TransformerStack(Module):
    """This version of Transformer acts both like an encoder and decoder."""

    def __init__(self, model_dim, nheads, ff_dim, num_layers, dropout=0.1,
                 mem_att_dropout=0.1, mem_att=True, mem_dim=None, prop_dim=None):
        super(TransformerStack, self).__init__()
        layer = TransformerDecoderLayer(model_dim=model_dim, nheads=nheads,
                                        ff_dim=ff_dim, dropout=dropout,
                                        mem_att=mem_att, mem_dim=mem_dim,
                                        mem_att_dropout=mem_att_dropout)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = LayerNorm(model_dim)
        self.tgt_mask = None

        self.prop_dim = prop_dim if prop_dim is not None and prop_dim > 0 else None

        if self.prop_dim is not None:
            self.inp_proj = Sequential()
            self.inp_proj.add_module("lin", Linear(model_dim + self.prop_dim,
                                                   model_dim))

    def forward(self, tgt, mem=None, tgt_key_padding_mask=None,
                mem_key_padding_mask=None, states=None, props=None, decode=True):
        """For more information see the Layer's documentation.

        Args:
            tgt (FloatTensor): the sequence to the decoder layer (required).
                [batch_size, tgt_seq_len, model_dim]
            mem (FloatTensor): the sequence from the last layer of the encoder
                (optional).
                [batch_size, src_seq_len, model_dim]
            tgt_key_padding_mask (ByteTensor): the mask for the tgt keys per
                batch (optional).
                [batch_size, tgt_seq_len]
            mem_key_padding_mask (ByTensor): the mask for the memory keys per
                batch (optional).
                [batch_size, mem_seq_len]
            states (FloatTensor): previously computed representations of the tgt
                of each layer.
                [num_layers, batch_size, state_seq_len, model_dim]
            props (FloatTensor): [batch_size, tgt_seq_len, prop_dim]
            decode (bool): whether it should use diagonal mask or no mask.
        Returns:
            out (FloatTensor): [batch_size, tgt_seq_len, model_dim]
            states (FloatTensor): [num_layers, batch_size, tgt_seq_len, model_dim]
            arts (dict):
                MEM_ATT_WTS (FloatTensor): [num_layers, batch_size, tgt_seq_len]
        """
        # conforming to the requirements of the downstream modules
        tgt = tgt.transpose(1, 0)
        mem = mem.transpose(1, 0) if mem is not None else None
        states = states.transpose(2, 1) if states is not None else None
        props = props.transpose(1, 0) if props is not None else None
        device = tgt.device

        # diagonal mask or none for decoding and encoding respectively
        if decode and \
                (self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt)):
            tgt_mask = _square_subs_mask(len(tgt)).to(device)
            self.tgt_mask = tgt_mask
        else:
            self.tgt_mask = None
        tgt_mask = self.tgt_mask

        # projecting props with input
        if props is not None and self.prop_dim is not None:
            tgt = self.inp_proj(T.cat((tgt, props), dim=-1))

        out = tgt
        new_states = []
        mem_att_wts_coll = []

        for i in range(self.num_layers):
            new_states.append(out)
            layer_state = None if states is None else states[i]
            layer = self.layers[i]
            out, mem_att_wts = layer(out, mem, tgt_mask=tgt_mask,
                                     mem_mask=None, state=layer_state,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     mem_key_padding_mask=mem_key_padding_mask)
            if mem_att_wts is not None:
                mem_att_wts_coll.append(mem_att_wts)

        if self.norm:
            out = self.norm(out)

        new_states.append(out)
        new_states = T.stack(new_states, dim=0)
        if len(mem_att_wts_coll):
            mem_att_wts_coll = T.stack(mem_att_wts_coll, dim=0)

        out = out.transpose(1, 0)
        new_states = new_states.transpose(2, 1)

        return out, new_states, {MEM_ATT_WTS: mem_att_wts_coll}


def _square_subs_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are
    filled with float('-inf'). Unmasked positions are filled with float(0.0).

    Returns:
        mask: [sz, sz]
    """
    mask = (T.triu(T.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf'))\
        .masked_fill(mask == 1, float(0.0))
    return mask
