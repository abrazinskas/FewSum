from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from logging import getLogger
import os

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class RatingProp(BaseTransformer):
    """Computes the rating deviation property for reviews. And
    that each batch contains data related to one group only!
    """

    def __init__(self, rating_fname, new_fname, **kwargs):
        super(RatingProp, self).__init__(**kwargs)
        self.rating_fname = rating_fname
        self.new_fname = new_fname

    def _transform(self, data_chunk):
        rating = data_chunk[self.rating_fname]
        data_chunk[self.new_fname] = []
        for indx in range(len(rating)):
            _hyp = rating[indx]
            _ref = [rating[i] for i in range(len(rating)) if i != indx]
            rating_dev = _comp_rating_dev(_hyp, _ref)
            data_chunk[self.new_fname].append(rating_dev)
        return data_chunk


def _comp_rating_dev(hyp_rating, refs_rating):
    res = hyp_rating - np.mean(refs_rating)
    return res
