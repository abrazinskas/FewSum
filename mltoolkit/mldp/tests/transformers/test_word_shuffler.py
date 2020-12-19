import unittest
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mldp.steps.transformers.nlp import WordShuffler
from mltoolkit.mlutils.helpers.formatting.seq import conv_seq_to_sent_symbols
import numpy as np
from copy import deepcopy
from .helpers import create_chunk

TEXT_FNAME = 'text'


class TestWordShuffler(unittest.TestCase):

    def test_condition_satisfaction(self):
        ks = [51.5, 2, 3, 4, 5]
        for k in ks:
            word_shuffler = WordShuffler(fname=TEXT_FNAME, end_symbol='DUMMY', k=k)
            dc = DataChunk(**{TEXT_FNAME: np.array([list(range(100)),
                                                    list(range(30))],
                                                   dtype='object')})
            corr_dc = word_shuffler(deepcopy(dc))

            for corr_du, du in zip(corr_dc.iter(), dc.iter()):
                text = du[TEXT_FNAME]
                corr_text = corr_du[TEXT_FNAME]
                self.assertTrue(condition_sat(corr_text, k=k))
                self.assertTrue(len(text) == len(corr_text))
                self.assertTrue(text != corr_text)

    def test_tokens_preservation(self):
        word_shuffler = WordShuffler(fname=TEXT_FNAME, end_symbol='.', k=3)
        batch_sizes = [1, 10, 20, 50]

        for batch_size in batch_sizes:
            dc = create_chunk(batch_size, TEXT_FNAME)
            corr_dc = word_shuffler(deepcopy(dc))

            for corr_du, du in zip(corr_dc.iter(), dc.iter()):
                text = du[TEXT_FNAME]
                corr_text = corr_du[TEXT_FNAME]
                self.assertTrue(all_tokens_preserved(text, corr_text))


def all_tokens_preserved(text, corr_text):
    text_sents = conv_seq_to_sent_symbols(text)
    corr_text_sents = conv_seq_to_sent_symbols(corr_text)
    if len(text_sents) != len(corr_text_sents):
        return False
    for sent, corr_sent in zip(text_sents, corr_text_sents):
        sent_counts = compute_counts(sent)
        corr_sent_counts = compute_counts(corr_sent)
        if len(sent_counts) != len(corr_sent_counts):
            return False

        for k in sent_counts:
            if k not in corr_sent_counts:
                return False
            if sent_counts[k] != corr_sent_counts[k]:
                return False
    return True


def compute_counts(seq):
    coll = {}
    for token in seq:
        if token not in coll:
            coll[token] = 0
        coll[token] += 1
    return coll


def condition_sat(corr_seq, k):
    for indx, i in enumerate(corr_seq):
        if not abs(indx - i) <= k:
            return False
    return True


if __name__ == '__main__':
    unittest.main()
