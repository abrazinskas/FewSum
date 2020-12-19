from mltoolkit.mldp.steps.transformers import BaseTransformer
from .pov_prop import compute_pov_distr
import numpy as np


class SummEvalPOVProp(BaseTransformer):
    """
    Computes POV by treating all summaries as one piece of text.
    Assumes that summaries are already tokenized. And that each batch contains
    data related to one group only!
    """

    def __init__(self, summ_fnames, new_fname, **kwargs):
        super(SummEvalPOVProp, self).__init__(**kwargs)
        self.summ_fnames = summ_fnames
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        assert data_chunk.valid
        coll = []
        for indx in range(len(data_chunk)):
            all_summ_toks = []
            for summ_fname in self.summ_fnames:
                summ_toks = data_chunk[indx, summ_fname]
                all_summ_toks += summ_toks
            coll.append(compute_pov_distr(all_summ_toks))
        data_chunk[self.new_fname] = coll
        return data_chunk
