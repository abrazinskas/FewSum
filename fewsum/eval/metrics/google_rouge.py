from mltoolkit.mlmo.eval.metrics import BaseMetric
from nltk import sent_tokenize
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .google_rouge_comps.rouge_scorer import RougeScorer


OLD_TO_NEW_METR_NAMES = {'precision': 'p', 'recall': 'r', 'fmeasure': 'f'}


class GoogleRouge(BaseMetric):
    def __init__(self, use_stemmer=True, remove_stopwords=False,
                 avg_rouge=True):
        super(GoogleRouge, self).__init__()
        self.remove_stopwords = remove_stopwords
        self.avg_rouge = avg_rouge
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                        use_stemmer=use_stemmer)
        self._rouge_coll = get_rouge_defaultdict(list)

    def accum(self, hyp, refs):
        """
        Args:
            hyp (list of str): Hypothesis (generated text)
                list of strs.
            refs (list of lists of str): Reference (gold) texts, where multiple
                instances are available to comparison against each hypothesis.
        Returns:
             list of dictionaries with scores
        """
        assert len(hyp) == len(refs)
        tmp_coll = []
        for _hyp, _refs in zip(hyp, refs):
            hyp_rouges = get_rouge_defaultdict(list)
            for _ref in _refs:
                scores = self._compute_rouge_scores(_hyp, _ref)

                # storing scores to the hypothesis score collector
                for rouge_name, rouge_obj in scores.items():
                    for old_name, new_name in OLD_TO_NEW_METR_NAMES.items():
                        score = getattr(rouge_obj, old_name)
                        hyp_rouges[rouge_name][new_name].append(score)

            # averaging or maximum over references
            for rouge_name, rouge_scores in hyp_rouges.items():
                for metr_name, metr_vals in rouge_scores.items():
                    val = np.mean(metr_vals) if self.avg_rouge \
                        else np.max(metr_vals)
                    hyp_rouges[rouge_name][metr_name] = val

            tmp_coll.append(hyp_rouges)
            self._update_collector(hyp_rouges)

        return tmp_coll

    def _compute_rouge_scores(self, hyp, ref):
        if self.remove_stopwords:
            ref = ' '.join([w for w in word_tokenize(ref)
                            if w not in self.stopwords])
            hyp = ' '.join([w for w in word_tokenize(hyp)
                            if w not in self.stopwords])
        rouges = self.rouge_scorer.score(ref, hyp)
        return rouges

    def aggr(self):
        """Aggregates results by computing the average over each rouge type.

        Returns: dict of dicts.
        """
        res = get_rouge_defaultdict(float)
        for rname, rouge_scores in self._rouge_coll.items():
            for mname, mvals in rouge_scores.items():
                res[rname][mname] = np.mean(mvals)
        return res

    def _update_collector(self, rouges_scores):
        """Updates the collector by storing current rouge scores."""
        for rouge_name, rouge_scores in rouges_scores.items():
            for metric_name, metr_score in rouge_scores.items():
                self._rouge_coll[rouge_name][metric_name].append(metr_score)


def get_rouge_defaultdict(default_type=float):
    """Returns dict of default dicts."""
    dict = {'rouge1': defaultdict(default_type),
            'rouge2': defaultdict(default_type),
            'rougeL': defaultdict(default_type)}
    return dict

