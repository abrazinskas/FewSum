from mltoolkit.mldp.steps.transformers import BaseTransformer


class TextLengthFilter(BaseTransformer):
    """Selects a fixed number of tokens from a tokenized text field."""

    def __init__(self, fname, max_len, **kwargs):
        super(TextLengthFilter, self).__init__(**kwargs)
        assert isinstance(max_len, int) and max_len > 0
        self.fname = fname
        self.max_len = max_len

    def _transform(self, data_chunk):
        for du in data_chunk.iter():
            du[self.fname] = du[self.fname][:self.max_len]
        return data_chunk
