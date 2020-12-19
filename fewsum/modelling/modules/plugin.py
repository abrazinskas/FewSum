from fewsum.modelling.modules.transformer_stack import MEM_ATT_WTS, TransformerStack
from fewsum.utils.fields import ModelF
from torch.nn import Module, Sequential, Linear, Sigmoid, Softmax, ModuleDict
import torch as T


class Plugin(Module):
    """Module used to compute prop values that are passed to the decoder."""

    def __init__(self, model_dim, ff_dim, nheads, nlayers, dropout=0.0,
                 mem_att_dropout=0.0, mem_dim=None,
                 len_prop=True, rating_prop=True, rouge_prop=True,
                 pov_prop=True):
        super(Plugin, self).__init__()
        assert len_prop or rating_prop or rouge_prop or pov_prop
        self.model_dim = model_dim
        self.tr_stack = TransformerStack(model_dim=model_dim, mem_dim=mem_dim,
                                         num_layers=nlayers, dropout=dropout,
                                         mem_att_dropout=mem_att_dropout,
                                         ff_dim=ff_dim, nheads=nheads)
        self.fns = ModuleDict()
        if len_prop:
            self.fns[ModelF.LEN_PROP] = Linear(model_dim, 1)
        if rating_prop:
            self.fns[ModelF.RATING_PROP] = Linear(model_dim, 1)
        if rouge_prop:
            self.fns[ModelF.ROUGE_PROP] = Sequential()
            self.fns[ModelF.ROUGE_PROP].add_module('lin', Linear(model_dim, 3))
            self.fns[ModelF.ROUGE_PROP].add_module('sigmoid', Sigmoid())
        if pov_prop:
            self.fns[ModelF.POV_PROP] = Sequential()
            self.fns[ModelF.POV_PROP].add_module("lin", Linear(model_dim, 4))
            self.fns[ModelF.POV_PROP].add_module('softmax', Softmax(dim=-1))

    def forward(self, mem, mem_bin_mask):
        """Computes props by running one-step transformer.

        Args:
            mem: [batch_size, seq_len, model_dim]
            mem_bin_mask: [batch_size, seq_len]

        Returns:
            dict mapping prop names to computed values [batch_size, dim].
        """
        bs = mem.size(0)
        dummy_inp = T.ones((bs, 1, self.model_dim), device=mem.device)
        out, _, tr_arts = self.tr_stack(tgt=dummy_inp, mem=mem,
                                        mem_key_padding_mask=mem_bin_mask)
        out = out.squeeze(1)
        props = {n: fn(out).squeeze(-1) for n, fn in self.fns.items()}
        mem_att_wts = tr_arts[MEM_ATT_WTS]
        return props, mem_att_wts
