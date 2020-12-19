from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from fewsum.utils.helpers.computation import compute_pov_distr


class POVProp(BaseTransformer):
    """Computes text POVs based on pronoun counts. Assumes tokenized text. And
    that each batch contains data related to one group only!
    """

    def __init__(self, text_fname, new_fname, **kwargs):
        super(POVProp, self).__init__(**kwargs)
        self.text_fname = text_fname
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        text = data_chunk[self.text_fname]
        povs = []
        for _text in text:
            assert isinstance(_text, (list, np.ndarray))
            povs.append(compute_pov_distr(_text))
        data_chunk[self.new_fname] = povs
        return data_chunk


