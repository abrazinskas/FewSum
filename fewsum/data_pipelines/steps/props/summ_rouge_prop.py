from mltoolkit.mldp.steps.transformers import BaseTransformer
from fewsum.eval.metrics import GoogleRouge


class SummRougeProp(BaseTransformer):
    """Summary ROUGE computation, where one summary is used as hypothesis."""

    def __init__(self, summ_fname, rev_fname, new_fname,
                 summ_group_indx_fname, group_rev_indxs_fname, rouge_kwargs=None,
                 **kwargs):
        super(SummRougeProp, self).__init__(**kwargs)
        rouge_kwargs = rouge_kwargs if rouge_kwargs is not None else {}
        self.summ_fname = summ_fname
        self.rev_fname = rev_fname
        self.new_fname = new_fname
        self.summ_group_indx_fname = summ_group_indx_fname
        self.group_rev_indxs_fname = group_rev_indxs_fname
        self._rouge = GoogleRouge(**rouge_kwargs)

    def _transform(self, data_chunk):
        data_chunk[self.new_fname] = []
        for i, summ_group_indx in enumerate(data_chunk[self.summ_group_indx_fname]):
            summ = data_chunk[i, self.summ_fname]
            assert isinstance(summ, str)
            _group_rev_indxs = data_chunk[summ_group_indx, self.group_rev_indxs_fname]
            revs = [data_chunk[indx, self.rev_fname] for indx in _group_rev_indxs]
            for rev in revs:
                assert isinstance(rev, str)
            scores = self._rouge.accum(hyp=[summ], refs=[revs])[0]
            data_chunk[self.new_fname].append([scores[n]['f'] for n in
                                               ['rouge1', 'rouge2', 'rougeL']])
        return data_chunk

