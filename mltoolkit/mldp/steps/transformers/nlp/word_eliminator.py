from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mlutils.helpers.general import listify


class WordEliminator(BaseTransformer):
    """
    Eliminates specified words from sequences. E.g. can eliminate <UNK>
    symbols.
    """
    
    def __init__(self, fname, words, **kwargs):
        """
        :param fname: name(s) of fields that contain symbols.
        :param words: symbol(str/int) or list of str/int to eliminate.
        """
        super(WordEliminator, self).__init__(**kwargs)
        self.fnames = listify(fname)
        self.words = set(words) if isinstance(words, list) else {words}
        
    def _transform(self, data_chunk):
        for fname in self.fnames:
            for indx in range(len(data_chunk)):
                elim_words(data_chunk[indx, fname], self.words)
        return data_chunk


def elim_words(seq, words_to_elim):
    i = 0
    while i < len(seq):
        word = seq[i]
        if word in words_to_elim:
            del seq[i]
        else:
            i += 1
