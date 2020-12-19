from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.constants.dp import EMPTY_CHUNK


class TextLenFilter(BaseTransformer):
    """Drops units that are longer or shorter than the threshold."""

    def __init__(self, fname, min_len=None, max_len=None, **kwargs):
        super(TextLenFilter, self).__init__(**kwargs)
        if min_len is not None:
            assert isinstance(min_len, int) and min_len > 0
        if max_len is not None:
            assert isinstance(max_len, int) and max_len > 0
        if min_len is not None and max_len is not None:
            assert min_len <= max_len
        self.fname = fname
        self.min_len = min_len
        self.max_len = max_len

    def _transform(self, data_chunk):
        assert data_chunk.valid
        for indx in reversed(range(len(data_chunk))):
            text = data_chunk[indx, self.fname]
            text_len = len(text)
            if (self.min_len is not None and text_len < self.min_len) or \
                    (self.max_len is not None and text_len > self.max_len):
                del data_chunk[indx]
        if len(data_chunk) == 0:
            return EMPTY_CHUNK
        return data_chunk
