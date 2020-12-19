import unittest
from .helpers import create_chunk
from mltoolkit.mldp.steps.transformers.nlp import HierWordDropper
from copy import deepcopy
import numpy as np

TEXT = 'text'
np.random.seed(42)


class TestHierWordDropper(unittest.TestCase):

    def test_dropping_all(self):
        corr_prob = 1.
        beta_a = 100000.
        beta_b = 0.000000001
        batch_size = 1000
        hwd = HierWordDropper(fname=TEXT, corr_prob=corr_prob, beta_a=beta_a,
                              beta_b=beta_b)

        dc = create_chunk(size=batch_size, fname=TEXT)

        corr_dc = hwd(deepcopy(dc))

        for du in corr_dc.iter():
            corr_text = du[TEXT]
            self.assertTrue(len(corr_text) == 0)

    def test_dropping_none(self):
        corr_prob = 0.
        beta_a = 5
        beta_b = 5
        batch_size = 1000
        hwd = HierWordDropper(fname=TEXT, corr_prob=corr_prob, beta_a=beta_a,
                              beta_b=beta_b)

        dc = create_chunk(size=batch_size, fname=TEXT)

        corr_dc = hwd(deepcopy(dc))

        for du, corr_du in zip(dc.iter(), corr_dc.iter()):
            text = du[TEXT]
            corr_text = corr_du[TEXT]
            self.assertTrue(len(corr_text) == len(text))

    def test_dropping_none2(self):
        corr_prob = 1.
        beta_a = 0.1
        beta_b = 500000
        batch_size = 1000
        hwd = HierWordDropper(fname=TEXT, corr_prob=corr_prob, beta_a=beta_a,
                              beta_b=beta_b)

        dc = create_chunk(size=batch_size, fname=TEXT)

        corr_dc = hwd(deepcopy(dc))

        for du, corr_du in zip(dc.iter(), corr_dc.iter()):
            text = du[TEXT]
            corr_text = corr_du[TEXT]
            self.assertTrue(len(corr_text) == len(text))

    def test_not_dropping_excl_symbol(self):
        """The transformer should never drop excluded symbols."""
        excl_symbol = '.'
        corr_prob = 1.
        beta_a = 5.0
        beta_b = 8.5
        batch_size = 1000
        hwd = HierWordDropper(fname=TEXT, corr_prob=corr_prob, beta_a=beta_a,
                              beta_b=beta_b, excl_symbols={excl_symbol})
        dc = create_chunk(size=batch_size, fname=TEXT)
        corr_dc = hwd(deepcopy(dc))

        for du, corr_du in zip(dc.iter(), corr_dc.iter()):
            text = du[TEXT]
            corr_text = corr_du[TEXT]

            self.assertTrue(len(corr_text) < len(text))

            exp_excl_symb_count = sum([1 for t in text if t == excl_symbol])
            act_excl_symb_count = sum([1 for t in corr_text if t == excl_symbol])

            self.assertTrue(exp_excl_symb_count == act_excl_symb_count)


if __name__ == '__main__':
    unittest.main()
