from mltoolkit.mldp.steps.transformers import BaseTransformer
from copy import deepcopy


class FieldDuplicator(BaseTransformer):
    """Duplicates fields by giving them new names."""

    def __init__(self, old_to_new_fnames, **kwargs):
        super(FieldDuplicator, self).__init__(**kwargs)
        self.old_to_new_fnames = old_to_new_fnames

    def _transform(self, data_chunk):
        for old_fn, new_fn in self.old_to_new_fnames.items():
            data_chunk[new_fn] = deepcopy(data_chunk[old_fn])
        return data_chunk
