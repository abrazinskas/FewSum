import unittest
from mltoolkit.mldp.tests.common import generate_data_chunk
from mltoolkit.mldp.steps.transformers import DataUnitMarker


class TestDataUnitsMarker(unittest.TestCase):

    def test_output(self):
        """Testing whether it produces a valid output."""
        new_fname = "DUMMY"
        dc = generate_data_chunk(10, 1000)
        th = 0.1
        dtypes = ['int64', 'int32', 'float32', 'float64', 'bool']
        key_fname = list(dc.keys())[0]
        eval_func = lambda x: x[key_fname] > th

        for dtype in dtypes:
            exp_fvals = (dc[key_fname] > th).astype(dtype)
            dum = DataUnitMarker(new_fname=new_fname, eval_func=eval_func,
                                 dtype=dtype)
            dc = dum(dc)
            self.assertTrue((dc[new_fname] == exp_fvals).all())



if __name__ == '__main__':
    unittest.main()
