from mltoolkit.mldp.steps.formatters import BaseFormatter
import numpy as np


class NumpyFormatter(BaseFormatter):
    def __init__(self, fnames, **kwargs):
        super(NumpyFormatter, self).__init__(**kwargs)
        if not isinstance(fnames, list):
            fnames = [fnames]
        self.fnames = fnames

    def _format(self, data_chunk):
        for fn in self.fnames:
            data_chunk[fn] = np.array(data_chunk[fn])
        return data_chunk
