from mltoolkit.mlmo.interfaces import ITorchModel
from logging import getLogger
import torch as T
from mltoolkit.mlmo.utils.tools import DecState
from fewsum.utils.fields import ModelF

logger = getLogger(__name__)


class ISumm(ITorchModel):
    def __init__(self, gen_func, **kwargs):
        super(ISumm, self).__init__(**kwargs)
        self.gen_func = gen_func

    def generate(self, batch, min_seq_len, max_seq_len, use_true_props=True):
        """Generates/decodes sequences from a conditional pmf.

        Args:
            batch: self-explanatory.
            min_seq_len: minimum length of the generated sequence.
            max_seq_len: maximum allowed length for generated sequences.
            use_true_props: whether to use true property values or the ones
                inferred by the plug-in network.

        Returns:
            word_ids: list of word ids.
            prop_values: dict of lists.
        """
        self.model.eval()
        rev = batch[ModelF.REV].to(self.device)
        rev_mask = batch[ModelF.REV_MASK].to(self.device)

        group_rev_indxs = batch[ModelF.GROUP_REV_INDXS].to(self.device)
        group_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK].to(self.device)
        bs = group_rev_indxs.size(0)

        mem,\
        mem_bin_mask = self.model.create_mem(rev=rev, rev_mask=rev_mask,
                                             group_rev_indxs=group_rev_indxs,
                                             group_rev_indxs_mask=group_rev_indxs_mask)

        if use_true_props:
            len_prop = batch[ModelF.LEN_PROP].to(self.device)
            rating_prop = batch[ModelF.RATING_PROP].to(self.device)
            rouge_prop = batch[ModelF.ROUGE_PROP].to(self.device)
            pov_prop = batch[ModelF.POV_PROP].to(self.device)
            prop_vals = {ModelF.LEN_PROP: len_prop,
                            ModelF.RATING_PROP: rating_prop,
                            ModelF.ROUGE_PROP: rouge_prop,
                            ModelF.POV_PROP: pov_prop}
        else:
            if hasattr(self.model, 'plugin'):
                prop_vals, _ = self.model.plugin(mem, mem_bin_mask)
            else:
                prop_vals = {}

        dummy = T.zeros(group_rev_indxs.size(0), device=self.device)

        if min_seq_len is not None:
            min_lens = [min_seq_len] * bs
        else:
            min_lens = None

        with T.no_grad():
            init_dec_state = DecState(rec_vals={"dummy": dummy})
            word_ids, _ = self.gen_func(init_dec_state=init_dec_state,
                                        max_steps=max_seq_len,
                                        log_normalize=True,
                                        min_lens=min_lens,
                                        mem=mem, mem_bin_mask=mem_bin_mask,
                                        minimum=1, **prop_vals)
        prop_vals = {k: v.tolist() for k, v in prop_vals.items()}
        return word_ids, prop_vals

