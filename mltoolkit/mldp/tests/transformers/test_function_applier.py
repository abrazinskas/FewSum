import unittest
from mltoolkit.mldp.tests.common import generate_data_chunk, create_list_of_data_chunks
from mltoolkit.mldp.steps.transformers.general import FunctionApplier
import numpy as np


class TestFunctionApplier(unittest.TestCase):
    def test_output(self):
        data_size = 1234
        data_attrs_number = 15
        input_chunks_size = 10
        transform_attrs_number = 10

        functions = [lambda x: np.log(abs(x) + 1), lambda x: np.exp(x),
                     lambda x: x**2]
        data = generate_data_chunk(data_attrs_number, data_size)
        transform_attrs = list(data.keys())[:transform_attrs_number]
        input_data_chunks = create_list_of_data_chunks(data, input_chunks_size)

        for func in functions:
            function_applier = FunctionApplier({a:func for a in transform_attrs})
            for input_data_chunk in input_data_chunks:
                actual_chunk = function_applier(input_data_chunk)
                expected_chunk = input_data_chunk

                # transforming manually values of input data-chunks
                for transform_attr in transform_attrs:
                    expected_chunk[transform_attr] = \
                        func(expected_chunk[transform_attr])

                self.assertTrue(actual_chunk == expected_chunk)


if __name__ == '__main__':
    unittest.main()
