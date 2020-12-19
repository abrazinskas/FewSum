from mltoolkit.mldp.steps.transformers import BaseTransformer


class FieldMerger(BaseTransformer):
    """Merges values of multiple fields into one. Drops the merge fields."""

    def __init__(self, merge_fnames, new_fname, **kwargs):
        super(FieldMerger, self).__init__(**kwargs)
        self.merge_fnames = merge_fnames
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        res = [v for v in zip(*[data_chunk[fn] for fn in self.merge_fnames])]
        data_chunk[self.new_fname] = res
        for fn in self.merge_fnames:
            del data_chunk[fn]
        return data_chunk
