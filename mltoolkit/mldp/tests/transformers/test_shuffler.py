import unittest
from mltoolkit.mldp.steps.transformers.general import Shuffler
from mltoolkit.mldp.tests.common import generate_data_chunk, create_list_of_data_chunks
from itertools import product
from numpy import isclose
import copy
import numpy as np

np.random.seed(seed=10)


class TestShuffler(unittest.TestCase):
    
    def test_order_and_elements_presence(self):
        """
        Checks whether shuffler changes the order, and does not eliminate
        elements.
        """
        data_chunk_sizes = [30, 10, 100, 320]
        data_attrs_numbers = [2, 1, 3, 15]

        for data_chunk_size, data_attrs_number in product(data_chunk_sizes,
                                                          data_attrs_numbers):
            data_chunk = generate_data_chunk(data_attrs_number, data_chunk_size)
            original_data_chunk = copy.deepcopy(data_chunk)
            shuffler = Shuffler()

            # Checking if the order is actually broken for desired fields/attrs,
            # and all data-units are preserved for the shuffled fields
            shuffled_data_chunk = shuffler(data_chunk)

            for attr in shuffled_data_chunk.keys():
                res = isclose(original_data_chunk[attr],
                              shuffled_data_chunk[attr])
                self.assertFalse(res.all())

                res = isclose(sorted(original_data_chunk[attr]),
                              sorted(shuffled_data_chunk[attr]))
                self.assertTrue(res.all())


if __name__ == '__main__':
    unittest.main()
