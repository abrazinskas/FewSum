import unittest
from shared.data_pipeline.steps.transformers import WordDropper
from mltoolkit.mldp.utils.tools import DataChunk
import numpy as np
from copy import deepcopy

DUMMY_FNAME = 'dummy'
NEW_DUMMY_FNAME = 'new_dummy'


class WordDropperTest(unittest.TestCase):

    def testing_zero_corruption(self):
        data_dc = DataChunk(**{DUMMY_FNAME: np.array([range(10) for _ in range(5)])})

        exp_dc = DataChunk()
        exp_dc[DUMMY_FNAME] = deepcopy(data_dc[DUMMY_FNAME])
        exp_dc[NEW_DUMMY_FNAME] = np.zeros(len(data_dc), dtype='object')
        for indx in range(len(exp_dc)):
            exp_dc[indx, NEW_DUMMY_FNAME] = list(exp_dc[indx, DUMMY_FNAME])

        word_dropper = WordDropper(fname=DUMMY_FNAME,
                                   new_fname=NEW_DUMMY_FNAME, dropout_prob=0.)

        act_dc = word_dropper(data_dc)

        self.assertTrue(act_dc == exp_dc)

    def testing_absolute_corruption(self):
        data_dc = DataChunk(
            **{DUMMY_FNAME: np.array([range(10) for _ in range(5)])})

        exp_dc = DataChunk()
        exp_dc[DUMMY_FNAME] = deepcopy(data_dc[DUMMY_FNAME])
        exp_dc[NEW_DUMMY_FNAME] = np.zeros(len(data_dc), dtype='object')
        for indx in range(len(exp_dc)):
            exp_dc[indx, NEW_DUMMY_FNAME] = list()

        word_dropper = WordDropper(fname=DUMMY_FNAME,
                                   new_fname=NEW_DUMMY_FNAME, dropout_prob=1.)

        act_dc = word_dropper(data_dc)

        self.assertTrue(act_dc == exp_dc)

    def test_symbols_exclusion(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
