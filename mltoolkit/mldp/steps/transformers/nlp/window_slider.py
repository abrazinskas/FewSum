from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from mltoolkit.mldp.utils.helpers.validation import validate_field_names
from mltoolkit.mldp.utils.helpers.nlp.sequences import compute_windows
from mltoolkit.mldp.steps.transformers.nlp.helpers import create_new_field_name
from mltoolkit.mlutils.helpers.general import listify
import numpy as np


class WindowSlider(BaseTransformer):
    """
    Runs a rolling slider over a sequence. Creates a separate field for each
    field to which the slider was applied.

    Assumes 2D data, namely batch_size x sequences, where sequences can be
    of different sizes.
    """

    def __init__(self, field_names, window_size=5, step_size=1,
                 only_full_windows=False, new_window_field_name_suffix='window',
                 **kwargs):
        """
        :param field_names: str or list of str (str) corresponding to fields
                            which should be slided over.
        :param window_size: self-explanatory.
        :param step_size: self-explanatory.
        :param only_full_windows: if set to True guarantees that all windows
                                  will be of the same size.
        :param new_window_field_name_suffix: suffix for all newly created fields.
        """
        try:
            validate_field_names(field_names)
        except Exception as e:
            raise e

        super(WindowSlider, self).__init__(**kwargs)
        self.field_names = listify(field_names)
        self.window_size = window_size
        self.step_size = step_size
        self.only_full_windows = only_full_windows
        self.new_windw_fn_suffix = new_window_field_name_suffix

    def _transform(self, data_chunk):
        for fn in self.field_names:
            field_vals = data_chunk[fn]
            tmp = np.empty(len(field_vals), dtype='object')
            for i, el in enumerate(field_vals):
                window_elms = compute_windows(el,
                                              window_size=self.window_size,
                                              step_size=self.step_size,
                                              only_full_windows=self.only_full_windows)
                tmp[i] = window_elms
            new_fn = create_new_field_name(fn, suffix=self.new_windw_fn_suffix)
            data_chunk[new_fn] = tmp
        return data_chunk
