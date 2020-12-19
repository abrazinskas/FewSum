import unittest
from fewsum.utils.helpers.computation import comp_cov_cmass
import torch as T


class TestHelpers(unittest.TestCase):

    def test_comp_cov_cmas(self):
        """"""
        probs = T.tensor([
            [
                [0.5, 0.2, 0.1, 0.05, 0.05],
                [0.8, 0.1, 0.05, 0.025, 0.025],
                [0.7, 0.2, 0.005, 0.005, 0.09]],
            [
                [0.1, 0.7, 0.1, 0.1, 0.1],
                [0.7, 0.1, 0.05, 0.05, 0.1],
                [0.1, 0.1, 0.3, 0.4, 0.1]
            ]
        ])  # [2, 3]

        lprobs = T.log(probs)
        words = T.tensor([
            [0, 1, 2, 0],
            [2, 3, 0, 0]
        ])  # [2, 4]
        words_mask = T.tensor([
            [1., 1., 1., 0.],
            [1., 1., 0., 0.]
        ])
        exp_out = T.tensor([[0.8, 0.95, 0.905],
                            [0.2, 0.1, 0.7]])

        act_out = comp_cov_cmass(lprobs, words, words_mask)

        self.assertTrue(T.all(T.isclose(exp_out, act_out)))


if __name__ == '__main__':
    unittest.main()
