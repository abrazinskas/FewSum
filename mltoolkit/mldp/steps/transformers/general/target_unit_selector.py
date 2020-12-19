from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.constants.dp import EMPTY_CHUNK
import numpy as np


class TargetUnitSelector(BaseTransformer):
    """
    Selects only marked data-units based on a binary field (target_indxs_fn).
    Drops target_indxs_fn in the end of the process.
    """

    def __init__(self, target_indic_fname, new_original_indxs_fname=None,
                 **kwargs):
        """
        :param target_indic_fname: binary indicators field name marking data
                                   units.
        :param new_original_indxs_fname: if provided, will create a new field with
                                  original indxs of data-units.
        """
        super(TargetUnitSelector, self).__init__(**kwargs)
        self.target_indic_fname = target_indic_fname
        self.new_original_indxs_fname = new_original_indxs_fname

    def _transform(self, data_chunk):
        if data_chunk[self.target_indic_fname].dtype not in [np.int32, np.int64]:
            raise ValueError("Target indxs field should contain int 0 and 1 "
                             "values only.")
        if self.new_original_indxs_fname:
            data_chunk[self.new_original_indxs_fname] = np.arange(len(data_chunk),
                                                                  dtype='int64')
        mask = data_chunk[self.target_indic_fname] == 1
        for fn in data_chunk:
            data_chunk[fn] = data_chunk[fn][mask]

        del data_chunk[self.target_indic_fname]

        # in case of no data-units were marked, return an empty chunk
        if len(data_chunk) == 0:
            return EMPTY_CHUNK

        return data_chunk
