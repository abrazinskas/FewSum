from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mlutils.helpers.general import listify


class FieldDropper(BaseTransformer):

    def __init__(self, fnames, **kwargs):
        """
        :param fnames: field(s) to be removed from chunks. Can be a list or str.
        """
        super(FieldDropper, self).__init__(**kwargs)
        self.fnames = listify(fnames)

    def _transform(self, data_chunk):
        for fname in self.fnames:
            del data_chunk[fname]
        return data_chunk
