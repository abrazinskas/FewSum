from mltoolkit.mldp.steps.transformers import BaseTransformer
from fewsum.eval.metrics import GoogleRouge


class RougeProp(BaseTransformer):
    """ROUGE property - one vs rest. Creates new fields with ROUGE scores.
    Assumes  that each batch contains data related to one group only!
    """

    def __init__(self, rev_fname, rouge_kwargs=None, **kwargs):
        super(RougeProp, self).__init__(**kwargs)
        rouge_kwargs = rouge_kwargs if rouge_kwargs is not None else {}
        self.rev_fname = rev_fname
        self._rouge = GoogleRouge(**rouge_kwargs)

    def _transform(self, data_chunk):
        rev = data_chunk[self.rev_fname]
        for indx in range(len(rev)):
            _hyp = rev[indx]
            _ref = [rev[i] for i in range(len(rev)) if i != indx]
            rouge_scores = self._rouge.accum(hyp=[_hyp], refs=[_ref])[0]
            for r_name, r_val in rouge_scores.items():
                if r_name not in data_chunk:
                    data_chunk[r_name] = []
                data_chunk[r_name].append(r_val['f'])
        return data_chunk
