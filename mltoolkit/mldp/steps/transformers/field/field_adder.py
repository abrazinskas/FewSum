from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class FieldAdder(BaseTransformer):
    """Adds an extra field to data-chunks with a fixed value."""

    def __init__(self, fname, fvalue, **kwargs):
        super(FieldAdder, self).__init__(**kwargs)
        self.fname = fname
        self.fvalue = fvalue

    def _transform(self, data_chunk):
        assert self.fname not in data_chunk
        data_chunk[self.fname] = np.full(shape=len(data_chunk),
                                         fill_value=self.fvalue)
        return data_chunk
