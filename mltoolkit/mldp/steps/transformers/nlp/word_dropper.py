from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class WordDropper(BaseTransformer):
    """
    Randomly drops words (tokens or ids) from tokenized sequences. Allows to
    exclude certain symbols from being dropped, e.g. end of sentences (.).

    Notice that it will convert each sequence to list, and operation is
    performed in-place.
    """

    def __init__(self, fname, dropout_prob, excluded_symbols=None,
                 substitute=None, **kwargs):
        """
        :param fname: field name of the sequences words of which should be
                      dropped.
        :param dropout_prob: self-explanatory.
        :param excluded_symbols: symbols that should not be dropped. E.g. '.'
        :param substitute: if provided, will replace dropped words with it.
        """
        super(WordDropper, self).__init__(**kwargs)
        assert dropout_prob <= 1.
        assert dropout_prob >= 0.
        self.fname = fname
        self.dropout_prob = dropout_prob
        self.excluded_symbols = excluded_symbols if excluded_symbols else {}
        self.substitute = substitute

    def _transform(self, data_chunk):
        if self.dropout_prob > 0.:
            field = np.zeros(len(data_chunk), dtype=object)
            for indx in range(len(data_chunk)):
                corr_seq = drop_words(seq=data_chunk[indx, self.fname],
                                      excl_symbols=self.excluded_symbols,
                                      dropout_prob=self.dropout_prob,
                                      substitute=self.substitute)
                field[indx] = corr_seq
            data_chunk[self.fname] = field
        return data_chunk


def drop_words(seq, dropout_prob, excl_symbols=None, substitute=None):
    """Drops words based on the parameters or optionally substitutes them."""
    excl_symbols = excl_symbols if excl_symbols else {}
    assert isinstance(seq, (list, np.ndarray))
    drop_symbols = np.random.binomial(1, p=dropout_prob, size=len(seq))
    corr_seq = []
    for symbol, drop_symbol in zip(seq, drop_symbols):
        if symbol in excl_symbols:
            corr_seq.append(symbol)
            continue
        if drop_symbol == 0:
            corr_seq.append(symbol)
        else:
            if substitute:
                corr_seq.append(substitute)
    return corr_seq
