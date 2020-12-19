import unittest
from mltoolkit.mldp.steps.transformers.nlp import WindowSlider
from mltoolkit.mldp.steps.transformers.nlp.helpers import create_new_field_name
from mltoolkit.mldp.utils.tools import DataChunk
import numpy as np


class TestWindowSlider(unittest.TestCase):
    def setUp(self):
        self.field_name = "dummy"
        self.suffix = "window"
        self.new_field_name = create_new_field_name(self.field_name,
                                                    suffix=self.suffix)

    # TODO: more descriptive method names would be nice to have
    
    def test_scenario1(self):
        window_size = 2
        step_size = 1
        only_full_windows = False
        input_seqs = np.array([list(range(6)), list(range(2))])
        input_chunk = DataChunk(**{self.field_name: input_seqs})
        expect_seqs = np.array([
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5]],
            [[0, 1]]])
        expected_output_chunk = DataChunk(**{self.field_name: input_seqs,
                                           self.new_field_name: expect_seqs})
        self._test_window_setup(input_chunk, expected_output_chunk,
                                field_name=self.field_name, suffix=self.suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)
        
    def test_scenario2(self):
        window_size = 3
        step_size = 3
        only_full_windows = False
        input_seqs = np.array([list(range(7)), list(range(2))])
        input_chunk = DataChunk(**{self.field_name: input_seqs})
        expect_seqs = np.array([
            [[0, 1, 2], [3, 4, 5], [6]],
            [[0, 1]]])
        expected_output_chunk = DataChunk(**{self.field_name: input_seqs,
                                           self.new_field_name: expect_seqs})
        self._test_window_setup(input_chunk, expected_output_chunk,
                                field_name=self.field_name, suffix=self.suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)
    
    def test_scenario3(self):
        window_size = 3
        step_size = 10
        only_full_windows = False
        
        input_seqs = np.array([list(range(3)), list(range(2))])
        input_chunk = DataChunk(**{self.field_name: input_seqs})
        expect_seqs = np.empty(2, dtype="object")
        expect_seqs[0] = [[0, 1, 2]]
        expect_seqs[1] = [[0, 1]]
        expected_output_chunk = DataChunk(**{self.field_name: input_seqs,
                                           self.new_field_name: expect_seqs})

        self._test_window_setup(input_chunk, expected_output_chunk,
                                field_name=self.field_name, suffix=self.suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)
        
    def test_scenario4(self):
        window_size = 2
        step_size = 1
        only_full_windows = True
        
        input_seqs = np.array([list(range(6)), list(range(3)), list(range(1))])
        input_chunk = DataChunk(**{self.field_name: input_seqs})
        expect_seqs = np.array([
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            [[0, 1], [1, 2]],
            []
        ])
        expected_output_chunk = DataChunk(**{self.field_name: input_seqs,
                                           self.new_field_name: expect_seqs})

        self._test_window_setup(input_chunk, expected_output_chunk,
                                field_name=self.field_name, suffix=self.suffix,
                                window_size=window_size, step_size=step_size,
                                only_full_windows=only_full_windows)

    def _test_window_setup(self, input_chunk, expected_output_chunk,
                           field_name, suffix,
                           window_size, step_size,
                           only_full_windows):
        window_slider = WindowSlider(field_names=field_name,
                                     window_size=window_size,
                                     step_size=step_size,
                                     new_window_field_name_suffix=suffix,
                                     only_full_windows=only_full_windows)
        actual_output_chunk = window_slider(input_chunk)

        self.assertTrue(expected_output_chunk == actual_output_chunk)


if __name__ == '__main__':
    unittest.main()
