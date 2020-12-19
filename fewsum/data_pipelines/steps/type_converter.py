from mltoolkit.mldp.steps.transformers import BaseTransformer
import os
from logging import getLogger

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class TypeConverter(BaseTransformer):
    """Converts field values to a specific types, removes invalid ones (opt)."""

    def __init__(self, fname, dtype_func, remove_invalid=True, **kwargs):
        super(TypeConverter, self).__init__(**kwargs)
        self.fname = fname
        self.dtype_func = dtype_func
        self.remove_invalid = remove_invalid

    def _transform(self, data_chunk):
        assert data_chunk.valid
        for indx in reversed(range(len(data_chunk))):
            try:
                val = data_chunk[indx, self.fname]
                data_chunk[indx, self.fname] = self.dtype_func(val)
            except Exception as e:
                if self.remove_invalid:
                    del data_chunk[indx]
                else:
                    raise e
        return data_chunk
