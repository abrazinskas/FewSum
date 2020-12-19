from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class DataUnitMarker(BaseTransformer):
    """
    Marks data-units with binary labels depending on the output of the provided
    evaluation function. Creates a separate field with the labels.
    """

    def __init__(self, new_fname, eval_func, dtype='int32', **kwargs):
        """
        :param new_fname: a name of the new field with binary labels.
        :param eval_func: a binary evaluation function of data-units. It should
                          input a data-unit and return either True or False.
        :param dtype: binary label's field's type. Allowed: 'int32', 'int64',
                      'float32', 'float64', 'bool'.
        """
        if not callable(eval_func):
            raise TypeError("Please provide a valid binary evaluation function.")
        valid_dtypes = ['int32', 'int64', 'float32', 'float64', 'bool']
        if dtype not in valid_dtypes:
            valid_dtypes_str = 'or '.join([str(dt) for dt in valid_dtypes])
            raise ValueError("Parameter 'dtype' must be %s." % valid_dtypes_str)
        super(DataUnitMarker, self).__init__(**kwargs)
        self.new_fname = new_fname
        self.eval_func = eval_func
        self.dtype = dtype

    def _transform(self, data_chunk):
        collector = np.zeros(len(data_chunk), dtype=self.dtype)
        for indx, du in enumerate(data_chunk.iter()):
            eval_res = self.eval_func(du)
            if isinstance(eval_res, bool):
                raise ValueError("The provided evaluation function returned the"
                                 " output of the type '%s', while it should be"
                                 " boolean only. " %
                                 type(eval_res))
            if eval_res:
                collector[indx] = 1
        data_chunk[self.new_fname] = collector
        return data_chunk
