from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mlutils.helpers.general import flatten
import numpy as np


class UniqueWordCalc(BaseTransformer):
    """Computes unique words per other reviews, used for the second loss."""

    def __init__(self, new_fname, rev_fname, other_rev_indxs_fname,
                 other_rev_indxs_mask_fname):
        super(UniqueWordCalc, self).__init__()
        self.new_fname = new_fname
        self.rev_fname = rev_fname
        self.other_rev_indxs_fname = other_rev_indxs_fname
        self.other_rev_inxs_mask_fname = other_rev_indxs_mask_fname

    def _transform(self, data_chunk):
        data_chunk[self.new_fname] = []

        rev = data_chunk[self.rev_fname]
        other_rev_indxs = data_chunk[self.other_rev_indxs_fname]
        other_rev_indxs_mask = data_chunk[self.other_rev_inxs_mask_fname]

        for indxs, mask in zip(other_rev_indxs, other_rev_indxs_mask):
            other_revs = [rev[indx] for indx, m in zip(indxs, mask) if m != 0]
            if isinstance(other_revs[0], np.ndarray):
                other_revs = np.concatenate(other_revs, axis=0)
            elif isinstance(other_revs[0], list):
                other_revs = flatten(other_revs)
            un_words = np.unique(other_revs)
            data_chunk[self.new_fname].append(un_words)
        return data_chunk
