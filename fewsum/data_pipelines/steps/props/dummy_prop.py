from mltoolkit.mldp.steps.transformers import BaseTransformer

# TODO: remove it?
class DummyProp(BaseTransformer):
    """Creates a dummy field filled with zeros."""

    def __init__(self, fname, new_fname, fval=0.):
        super(DummyProp, self).__init__()
        self.fname = fname
        self.new_fname = new_fname
        self.fval = fval

    def _transform(self, data_chunk):
        data_chunk[self.new_fname] = [self.fval] * len(data_chunk[self.fname])
        return data_chunk
