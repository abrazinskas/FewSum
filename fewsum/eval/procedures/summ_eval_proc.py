from fewsum.utils.fields import ModelF, OutDataF
from fewsum.eval.metrics import GoogleRouge
from logging import getLogger
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mldp.utils.tools import DataChunk
import codecs
import numpy as np
import os
from fewsum.utils.helpers.formatting import dct_list_to_list_dict, format_stats
from fewsum.utils.helpers.data import get_group_reviews

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)

ROUGE_KWARGS = {}


class SummEvalProc(object):
    """
    Performs evaluation on summaries based on the original Rouge.
    Writes results (true summs, gen summs, inp reviews, rouge scores) to a file.
    """

    def __init__(self, data_pipeline, summs_gen_func,
                 rev_formatter_func=None, summ_formatter_func=None,
                 analytics_func=None):
        """
        :param data_pipeline: self-explanatory.
        :param summs_gen_func: takes a batch and returns list of string
                               summaries.
        :param rev_formatter_func: takes reviews (field) and formats it
                                    by outputting list of strings.
        :param analytics_func: if provided will pass the generated summaries
                               though the function.
        """
        self.data_pipeline = data_pipeline
        self.summs_gen_func = summs_gen_func
        self.rev_formatter_func = rev_formatter_func
        self.summ_formatter_func = summ_formatter_func
        self.analytics_func = analytics_func

    def eval(self, data_source, out_file_path=None):
        """
        Assumes that batches contain SUMMS that are lists of sublists,
        where is sublist contain a fixed number of summary strings. I.e.
        summaries should not be tokenized.
        """
        output_dc = DataChunk(**{OutDataF.GOLD_SUMMS: [], OutDataF.GEN_SUMM: [],
                                 OutDataF.GROUP_ID: [], OutDataF.CAT: [],
                                 OutDataF.ROUGE: [], OutDataF.PROPS: [],
                                 OutDataF.PRED_PROPS: [],
                                 OutDataF.INP_REV: []})
        rouge_evaluator = GoogleRouge()

        prop_old_new_fnames = {ModelF.LEN_PROP: OutDataF.LEN_PROP,
                               ModelF.RATING_PROP: OutDataF.RATING_PROP,
                               ModelF.ROUGE_PROP: OutDataF.ROUGE_PROP,
                               ModelF.POV_PROP: OutDataF.POV_PROP}

        skipped_summs = 0
        gen_summs_coll = []

        for batch in self.data_pipeline.iter(**data_source):
            # notice that each product has K true summaries created by
            # annotators
            inp_revs = batch[ModelF.REV].numpy()
            true_summs = batch[ModelF.SUMMS]
            group_ids = list(batch[ModelF.GROUP_ID])
            cats = list(batch[ModelF.CAT])
            group_rev_indxs = batch[ModelF.GROUP_REV_INDXS].numpy()
            group_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK].numpy()

            gen_summ, pred_props = self.summs_gen_func(batch)
            assert (len(true_summs) == len(gen_summ))

            # below one will be used for analytics
            gen_summs_coll += gen_summ

            #  props
            props = {n: batch[o].tolist() for o, n
                     in prop_old_new_fnames.items()}
            props = dct_list_to_list_dict(props)
            output_dc[OutDataF.PROPS] += props
            if len(pred_props):
                pred_props = dct_list_to_list_dict(pred_props)
                output_dc[OutDataF.PRED_PROPS] += pred_props

            # accumulating ROUGE statistics
            rouge_scores = []
            for _gen_summ, _tr_summs in zip(gen_summ, true_summs):
                if len(_gen_summ) == 0:
                    skipped_summs += 1
                    rouge_scores.append(None)
                    continue

                # extra [] wrapping is needed as the accum method is batch based
                r_score = rouge_evaluator.accum(hyp=[_gen_summ], refs=[_tr_summs])[0]
                rouge_scores.append(r_score)

            # grouping and formatting
            if self.rev_formatter_func is not None:
                inp_revs = [self.rev_formatter_func(seq) for seq in inp_revs]
            group_revs = get_group_reviews(inp_revs, group_rev_indxs,
                                           group_rev_indxs_mask)

            if self.summ_formatter_func is not None:
                true_summs = [[self.summ_formatter_func(s) for s in _summs]
                              for _summs in true_summs]
                gen_summ = [self.summ_formatter_func(s) for s in gen_summ]

            # storing the output batch for later dumping
            output_dc[OutDataF.GOLD_SUMMS] += true_summs
            output_dc[OutDataF.GEN_SUMM] += gen_summ
            output_dc[OutDataF.INP_REV] += group_revs
            output_dc[OutDataF.CAT] += cats
            output_dc[OutDataF.GROUP_ID] += group_ids
            output_dc[OutDataF.ROUGE] += rouge_scores

        # some models don't output predicted props this condition deals with it
        if not len(output_dc[OutDataF.PRED_PROPS]):
            del output_dc[OutDataF.PRED_PROPS]

        # running analytics
        if self.analytics_func:
            formatted_true_summs = []
            for seq_coll in output_dc[OutDataF.GOLD_SUMMS]:
                for seq in seq_coll:
                    seq = seq if not isinstance(seq, list) else " ".join(seq)
                    formatted_true_summs.append(seq)
            an_scores = self.analytics_func(formatted_true_summs)
            form_an_scores = format_stats(an_scores, title="True Text Analytics")
            for s in form_an_scores:
                logger.info(s)

            an_scores = self.analytics_func(gen_summs_coll)
            form_an_scores = format_stats(an_scores, title="Gen. Text Analytics")
            for s in form_an_scores:
                logger.info(s)

        final_metrs = rouge_evaluator.aggr()
        form_final_metrs = format_stats(final_metrs, title="ROUGE Scores")
        for s in form_final_metrs:
            logger.info(s)

        if out_file_path:
            gr_fields = [OutDataF.CAT, OutDataF.GROUP_ID]
            safe_mkfdir(out_file_path)
            output_file = codecs.open(out_file_path, 'w')
            output_dc.to_json(f=output_file, grouping_fnames=gr_fields)
            logger.info("Wrote the eval output to: "
                        "'%s'." % out_file_path)
        logger.info("Not generated %d summaries." % skipped_summs)
