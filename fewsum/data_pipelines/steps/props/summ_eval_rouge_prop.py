from mltoolkit.mldp.steps.transformers import BaseTransformer
from fewsum.eval.metrics import GoogleRouge
import numpy as np


class SummEvalRougeKnob(BaseTransformer):
    """
    Summary specific ROUGE computation, where multiple summaries are used
    as hypotheses. Ignores None summaries (adjusted to filtering).
    Assumes that each batch contains data related to one group only!

    Used for evaluation.
    """

    def __init__(self, ref_fnames, hyp_fnames, new_fname, rouge_kwargs=None, **kwargs):
        super(SummEvalRougeKnob, self).__init__(**kwargs)
        rouge_kwargs = rouge_kwargs if rouge_kwargs is not None else {}
        if not isinstance(ref_fnames, list):
            ref_fnames = [ref_fnames]
        if not isinstance(hyp_fnames, list):
            hyp_fnames = [hyp_fnames]
        self.ref_fnames = ref_fnames
        self.hyp_fnames = hyp_fnames
        self.new_fname = new_fname
        self._rouge = GoogleRouge(**rouge_kwargs)

    def _transform(self, data_chunk):
        assert data_chunk.valid
        rouge_scores = []
        for indx in range(len(data_chunk)):
            _hyp = [data_chunk[indx, hyp_fname] for hyp_fname
                    in self.hyp_fnames]
            _ref = [data_chunk[indx, ref_fname] for ref_fname
                    in self.ref_fnames]
            _hyp = [_h for _h in _hyp if isinstance(_h, str)]  # ignores None summaries
            scores = self._rouge.accum(hyp=_hyp, refs=[_ref] * len(_hyp))[0]

            rouge_scores.append([scores[n]['f'] for n in
                                 ['rouge1', 'rouge2', 'rougeL']])
        data_chunk[self.new_fname] = rouge_scores

        return data_chunk

