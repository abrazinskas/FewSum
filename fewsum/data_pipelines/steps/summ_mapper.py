from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from mltoolkit.mldp.utils.constants.dp import EMPTY_CHUNK


class SummMapper(BaseTransformer):
    """
    Flattens the summaries, which assumed to be a list of lists to just a list.
    Creates a new field that maps them to the original indx.

    I decided not to alter AmazonTransformer, which in principle can be modified
    to achieve the same result. Because the mentioned step can be used afterwards
    for creation of the evaluation pipeline.
    """

    def __init__(self, fname, new_indx_fname, **kwargs):
        super(SummMapper, self).__init__(**kwargs)
        self.fname = fname
        self.new_indx_fname = new_indx_fname

    def _transform(self, data_chunk):
        new_summs, indxs = self._flatten(data_chunk[self.fname])
        if len(new_summs) == 0:
            return EMPTY_CHUNK
        data_chunk[self.fname] = new_summs
        data_chunk[self.new_indx_fname] = indxs
        return data_chunk

    def _flatten(self, summs):
        new_summ_list = []
        indxs = []
        for indx, _summs in enumerate(summs):
            new_summ_list += _summs
            indxs += [indx] * len(_summs)
        return new_summ_list, np.array(indxs)
