from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class LenProp(BaseTransformer):
    """Computes length deviation between reviews, i.e., one vs rest. Assumes that
    each batch contains data related to one group only!
    """

    def __init__(self, len_fname, new_fname, **kwargs):
        super(LenProp, self).__init__(**kwargs)
        self.len_fname = len_fname
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        length = data_chunk[self.len_fname]
        data_chunk[self.new_fname] = []
        for indx in range(len(length)):
            _hyp = length[indx]
            _ref = [length[i] for i in range(len(length)) if i != indx]
            len_dev = comp_len_dev(_hyp, _ref)
            data_chunk[self.new_fname].append(len_dev)
        return data_chunk


def comp_len_dev(hyp_len, refs_len):
    res = hyp_len - np.mean(refs_len)
    return res

