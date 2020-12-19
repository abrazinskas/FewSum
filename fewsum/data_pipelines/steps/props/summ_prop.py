from mltoolkit.mldp.steps.transformers import BaseTransformer


class SummProp(BaseTransformer):
    """
    An artificial prop-step that creates length, rating and POV.
    Shifts ROUGE that was created previously.
    """

    def __init__(self, rouge_prop_fname=None, len_prop_fname=None,
                 rating_prop_fname=None, pov_prop_fname=None,
                 len_prop_val=0., rating_prop_val=0., rouge_prop_shift=None,
                 pov_prop_val=None, **kwargs):
        super(SummProp, self).__init__(**kwargs)

        self.len_prop_fname = len_prop_fname
        self.rating_prop_fname = rating_prop_fname
        self.rouge_prop_fname = rouge_prop_fname
        self.pov_prop_fname = pov_prop_fname

        self.len_prop_val = len_prop_val
        self.rating_prop_val = rating_prop_val
        self.rouge_prop_shift = rouge_prop_shift if rouge_prop_shift is not None \
            else [0., 0., 0.]
        self.pov_distr_val = pov_prop_val if pov_prop_val is not None\
            else [0., 0., 1., 0.]

    def _transform(self, data_chunk):
        bs = len(data_chunk)

        if self.rouge_prop_fname:
            rouge_prop = data_chunk[self.rouge_prop_fname]
            # shifting rouge
            rouge_prop = [
                [_r_v + _r_s for _r_v, _r_s in zip(_rouge_prop, self.rouge_prop_shift)]
                for _rouge_prop in rouge_prop]
            data_chunk[self.rouge_prop_fname] = rouge_prop

        # artificially creating length, rating, and pov fields
        if self.len_prop_fname:
            data_chunk[self.len_prop_fname] = [self.len_prop_val] * bs
        if self.rating_prop_fname:
            data_chunk[self.rating_prop_fname] = [self.rating_prop_val] * bs
        if self.pov_prop_fname:
            data_chunk[self.pov_prop_fname] = [self.pov_distr_val] * bs

        return data_chunk
