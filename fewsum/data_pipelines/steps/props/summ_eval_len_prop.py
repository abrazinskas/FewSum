from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from .len_prop import comp_len_dev


class SummEvalLenProp(BaseTransformer):
    """
    Computes the average length deviation of summaries.
    Assuming that reviews and summaries are already tokenized. And
    that each batch contains data related to one group only!
    """

    def __init__(self, summ_fnames, rev_fnames, new_fname, **kwargs):
        super(SummEvalLenProp, self).__init__(**kwargs)
        self.summ_fnames = summ_fnames
        self.rev_fnames = rev_fnames
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        """
        Assumes that the data-chunk contains only one group reviews and
        summaries.
        """
        assert data_chunk.valid
        coll = []
        for indx in range(len(data_chunk)):
            summ_lens = [len(data_chunk[indx, summ_fname]) for summ_fname
                     in self.summ_fnames]
            rev_lens = [len(data_chunk[indx, rev_fname]) for rev_fname
                        in self.rev_fnames]
            avg_len_dev = 0.
            for summ_len in summ_lens:
                avg_len_dev += comp_len_dev(summ_len, rev_lens) / len(summ_lens)
            coll.append(avg_len_dev)
        data_chunk[self.new_fname] = coll
        return data_chunk
