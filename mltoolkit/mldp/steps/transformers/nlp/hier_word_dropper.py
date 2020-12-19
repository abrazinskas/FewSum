from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from mltoolkit.mlutils.helpers.general import listify


class HierWordDropper(BaseTransformer):
    """
    Hierarchical noise model that corrupts sentences with some probability.
    Corruption is based on a floating / dynamic word-dropout which is sampled
    from a beta distribution.
    """

    def __init__(self, fname, corr_prob, beta_a, beta_b, excl_symbols=None,
                 substitute=None, **kwargs):
        """
        :param fname: str or list of name strs of fields to which corruption
                      should be applied. 
        :param corr_prob: probability of corrupting a data-unit.
        :param beta_a: first parameter of the Beta distribution.
        :param beta_b: second parameter of the beta distribution.
        :param excl_symbols: a set of symbols that are never dropped.
        :param substitute: a symbol with which a dropped symbol should be 
                           replaced.
        """
        super(HierWordDropper, self).__init__(**kwargs)
        assert corr_prob <= 1.        
        self.fnames = listify(fname)
        self.corr_prob = corr_prob
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.excluded_symbols = excl_symbols if excl_symbols else {}
        self.substitute = substitute

    def _transform(self, data_chunk):
        if self.corr_prob > 0.:
            for fname in self.fnames:
                field = np.zeros(len(data_chunk), dtype=object)
                corr_flags = np.random.binomial(n=1, size=len(data_chunk),
                                                p=self.corr_prob)
                for indx, c_flag in zip(range(len(data_chunk)), corr_flags):
                    seq = data_chunk[indx, fname]
                    if c_flag:
                        seq = drop_words(seq=seq,
                                         excl_symbols=self.excluded_symbols,
                                         beta_a=self.beta_a, beta_b=self.beta_b,
                                         substitute=self.substitute)
                    field[indx] = seq
                data_chunk[fname] = field
        return data_chunk


def drop_words(seq, beta_a, beta_b, excl_symbols=None, substitute=None):
    """
    Drops words based on the beta parameters or optionally substitutes them.
    Samples a Bernoulli parameter for each token based on 'beta_a', and 'beta_b'
    .
    """
    excl_symbols = excl_symbols if excl_symbols else {}
    assert isinstance(seq, (list, np.ndarray))
    bern_params = np.random.beta(beta_a, beta_b, size=len(seq))
    drop_symbols = np.random.binomial(1, p=bern_params)
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
