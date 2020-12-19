from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mlutils.helpers.formatting.seq import conv_seq_to_sent_symbols
from mltoolkit.mlutils.helpers.general import argsort
import numpy as np


class WordShuffler(BaseTransformer):
    """
    Performs symbol shuffling as described in the unsupervised neural MT paper.
    Every permuted word satisfied the condition of being away from its original
    position not further than 'k'. Where 'k' is the passed hyper-parameter.

    Expects the passed sequences to be already tokenized.

    The step performs operation in-place, without creation of a separate field.
    Also, it performs internal split and merge by sentences, so multi sentence
    sequences are permitted.
    """

    def __init__(self, fname, k, end_symbol, **kwargs):
        """
        :param fname: name of field that contains sequences.
        :param k: shuffling hyper-parameter.
        :param end_symbol: symbol indicating the end of sentences, i.e. "."
        :param kwargs:
        """
        super(WordShuffler, self).__init__(**kwargs)
        self.fname = fname
        self.k = k
        self.end_symbol = end_symbol

    def _transform(self, data_chunk):
        if self.k >= 1.:
            field = np.zeros(len(data_chunk), dtype='object')
            for indx, du in enumerate(data_chunk.iter()):
                seq = du[self.fname]
                assert isinstance(seq, (list, np.ndarray))
                sents = conv_seq_to_sent_symbols(seq, end_symbol=self.end_symbol)
                corr_seq = []
                for indx2, sent in enumerate(sents):
                    corr_sent = self._shuffle_sent_symbols(sent)
                    # add the end symbol to the last sentence only if it
                    # originally had one
                    if indx2 == len(sents) - 1:
                        if seq[-1] == self.end_symbol:
                            corr_sent.append(self.end_symbol)
                    else:
                        corr_sent.append(self.end_symbol)
                    corr_seq += corr_sent
                field[indx] = corr_seq
            data_chunk[self.fname] = field
        return data_chunk

    def _shuffle_sent_symbols(self, sent):
        """Performs the actual shuffling using a random sampling process."""
        alpha = self.k + 1
        q = np.random.uniform(0, alpha, size=len(sent)) + np.arange(0, len(sent))
        perm = argsort(q, order='ascending')
        corr_sent = [sent[i] for i in perm]
        return corr_sent
