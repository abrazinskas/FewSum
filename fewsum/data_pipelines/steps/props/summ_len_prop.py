from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from .len_prop import comp_len_dev


class SummLenProp(BaseTransformer):
    """
    Summary specific length computation assuming that both summaries and
    reviews are tokenized already or that tokenization function is passed.
    """

    def __init__(self, summ_fname, rev_fname, new_fname,
                 summ_group_indx_fname, group_rev_indxs_fname, tok_func=None,
                 **kwargs):
        super(SummLenProp, self).__init__(**kwargs)
        self.summ_fname = summ_fname
        self.rev_fname = rev_fname
        self.new_fname = new_fname
        self.summ_group_indx_fname = summ_group_indx_fname
        self.group_rev_indxs_fname = group_rev_indxs_fname
        self.tok_func = tok_func

    def _transform(self, data_chunk):
        coll = []
        for summ, summ_grp_indx in zip(data_chunk[self.summ_fname],
                                       data_chunk[self.summ_group_indx_fname]):
            if isinstance(summ, str):
                if self.tok_func is None:
                    raise ValueError("Please provide a function to tokenize "
                                     "summaries.")
                summ = self.tok_func(summ)
            summ_len = len(summ)

            _grp_rev_indxs = data_chunk[summ_grp_indx, self.group_rev_indxs_fname]
            rev_lens = []
            for indx in _grp_rev_indxs:
                rev = data_chunk[indx, self.rev_fname]
                if isinstance(rev, str):
                    if self.tok_func is None:
                        raise ValueError("Please provide a function to tokenize "
                                         "reviews.")
                    rev = self.tok_func(rev)
                rev_lens.append(len(rev))

            dev = comp_len_dev(summ_len, rev_lens)
            coll.append(dev)
        data_chunk[self.new_fname] = coll
        return data_chunk

