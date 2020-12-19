import unittest
from fewsum.eval.metrics.google_rouge import GoogleRouge, OLD_TO_NEW_METR_NAMES
from fewsum.eval.metrics.google_rouge_comps.rouge_scorer import RougeScorer
from numpy import isclose
from collections import defaultdict


class TestGoogleRouge(unittest.TestCase):
    """Tests correctness of my modified wrapper."""

    def test_avg_rouge(self):
        """Checks if my global average is correctly implemented."""

        hyp1 = 'Dog is brown'
        ref1 = "dog is Brown"

        hyp2 = "Computer is very fast, great!"
        ref2 = "Computer is super fast!"

        hyp3 = "Shipping was very fast"
        ref3 = "Very fast shipping"

        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        eval_metric = GoogleRouge()
        eval_metric.accum([hyp1, hyp2, hyp3],
                          [[ref1], [ref2], [ref3]])
        actual_scores = eval_metric.aggr()

        # computing expected rouge scores using the default Rouge scorer
        # (not wrapper)
        expected_scores =  {'rouge1': defaultdict(float),
                            'rouge2': defaultdict(float),
                            'rougeL': defaultdict(float)}
        hyps = [hyp1, hyp2, hyp3]
        refs = [ref1, ref2, ref3]
        for hyp, ref in zip(hyps, refs):
            scores = rouge_scorer.score(ref, hyp)

            for rname, rvals in scores.items():
                for mname, new_mname in OLD_TO_NEW_METR_NAMES.items():
                    mval = getattr(rvals, mname)
                    if new_mname not in expected_scores[rname]:
                        expected_scores[rname][new_mname] = 0.
                    expected_scores[rname][new_mname] += mval / len(hyps)

        # compering the actual and the expected rouge scores
        for rname in expected_scores:
            self.assertTrue(
                expected_scores[rname].keys() == actual_scores[rname].keys())
            for mname in expected_scores[rname]:
                act_mval = actual_scores[rname][mname]
                exp_mval = expected_scores[rname][mname]
                self.assertTrue(isclose(act_mval, exp_mval))


if __name__ == '__main__':
    unittest.main()
