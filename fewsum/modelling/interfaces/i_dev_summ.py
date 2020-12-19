from mltoolkit.mlutils.helpers.formatting.general import format_big_box, stats_to_str
from mltoolkit.mldp.utils.tools import DataChunk
from fewsum.utils.fields import ModelF, OutDataF
from mltoolkit.mlmo.interfaces import IBaseDevTorch
import codecs
from logging import getLogger
import os
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir, comb_paths,\
    get_file_name, safe_mkdir
from fewsum.utils.helpers.formatting import format_stats
from fewsum.eval.procedures import SummEvalProc
from functools import partial
from fewsum.utils.helpers.formatting import dct_list_to_list_dict
from time import time
from collections import OrderedDict
from torch.cuda import empty_cache
from fewsum.utils.helpers.stats import accum_stats

logger_name = os.path.basename(__name__)
logger = getLogger(logger_name)


class IDevSumm(IBaseDevTorch):
    """Development interface containing high-level logic for training and
    evaluating the summarizer also for summary generation.
    """

    def __init__(self, word_vocab, seq_postproc=None, summ_postproc=None,
                 eval_data_pipeline=None, seq_gen_data_pipeline=None,
                 seq_analytics=None, **kwargs):
        """
        Args:
            word_vocab: vocabulary with words.
            seq_postproc: a function that converts generated sequences to a
                visualiazble format.
            summ_postproc:  a function that converts string sequences to a
                visualiazble format.
            eval_data_pipeline: data pipeline that yields batches for
                summarization and evaluation
            seq_gen_data_pipeline: data pipeline that yields batches for
                leave-one-out review generation.
            seq_analytics: function for text analytics.
        """
        super(IDevSumm, self).__init__(**kwargs)
        self.word_vocab = word_vocab
        self.gen_seq_postproc = seq_postproc
        self.summ_postproc = summ_postproc
        self.eval_pipeline = eval_data_pipeline
        self.seq_gen_pipeline = seq_gen_data_pipeline
        self.seq_analytics = seq_analytics

    def standard_workflow(self, train_data_source, val_data_source=None,
                          test_data_source=None, epochs=1,
                          logging_period=10, eval_train=False,
                          after_epoch_func=None):
        """
        Runs a workflow of steps such as training and evaluation. It executes
        a very general workflow, where eval flags can be assigned in order to
        perform evaluation on train, val, and test data-sources.

        :param train_data_source: self-explanatory.
        :param val_data_source: self-explanatory.
        :param test_data_source: self-explanatory.
        :param epochs: self-explanatory.
        :param logging_period: how often to log the loss of the model
        :param eval_train: whether to eval performance on the training
                           data-source.
        :param after_epoch_func: a function that takes as input 'epoch', and is
                                 executed after completion of each epoch, except
                                 the last one. E.g. model saving.
        """
        if not isinstance(val_data_source, list):
            val_data_source = [val_data_source]

        for val_ds in val_data_source:
            stats = self.eval(data_source=val_ds)
            logger.info(stats_to_str(stats, "Validation"))

        epoch = 0
        for epoch in range(1, epochs + 1):
            logger.info('Epoch %d/%d' % (epoch, epochs))
            self.train(data_source=train_data_source, epoch=epoch,
                       logging_steps=logging_period)

            for val_ds in val_data_source:
                stats = self.eval(data_source=val_ds)
                logger.info(stats_to_str(stats, "Validation"))

            if after_epoch_func:
                after_epoch_func(epoch)

        if test_data_source:
            stats = self.eval(data_source=test_data_source,
                              epoch=epoch)
            logger.info(stats_to_str(stats, "Testing"))

    def summ_eval(self, out_dir_path, data_source, **kwargs):
        """Runs evaluation of summaries."""
        assert self.eval_pipeline is not None
        empty_cache()
        summ_gen_func = partial(self.summ_gen_wrapper, **kwargs)
        output_fn = "%s_eval.json" % get_file_name(
            data_source['data_path'])
        out_file_path = comb_paths(out_dir_path, output_fn)
        logger.info("Performing summary evaluation on %s." % data_source)
        eval_proc = SummEvalProc(self.eval_pipeline, summs_gen_func=summ_gen_func,
                                 rev_formatter_func=self.gen_seq_postproc,
                                 summ_formatter_func=self.summ_postproc,
                                 analytics_func=self.seq_analytics)
        eval_proc.eval(data_source, out_file_path=out_file_path)

    def gen_seqs(self, data_source, out_file_path, **kwargs):
        """Generates sequences and saves them to a file."""
        assert self.seq_gen_pipeline is not None
        empty_cache()

        safe_mkfdir(out_file_path)
        logger.info(f'Generating conditional sequences/summaries for '
                    f'{data_source}.')

        # storing to a data-chunk and dumping to the storage
        output_dc = DataChunk()
        fields_to_init = [OutDataF.GROUP_ID, OutDataF.INP_REV, OutDataF.GEN_REV,
                          OutDataF.REV_INDX, OutDataF.INP_REV_RATING,
                          OutDataF.INP_REV_LEN, OutDataF.GEN_REV_LEN,
                          OutDataF.PROPS, OutDataF.PRED_PROPS]
        for fname in fields_to_init:
            output_dc[fname] = []

        prop_old_new_fnames = {ModelF.LEN_PROP: OutDataF.LEN_PROP,
                               ModelF.RATING_PROP: OutDataF.RATING_PROP,
                               ModelF.ROUGE_PROP: OutDataF.ROUGE_PROP,
                               ModelF.POV_PROP: OutDataF.POV_PROP}

        for batch in self.seq_gen_pipeline.iter(**data_source):

            rev_group_id = batch[ModelF.REV_GROUP_ID]
            rev_indx = _comp_rev_indx(rev_group_id)
            rating = batch[ModelF.REV_RATING]
            inp_rev = batch[ModelF.REV].numpy()

            gen_rev, pred_props = self.imodel.generate(batch=batch, **kwargs)

            # post-processing
            if self.gen_seq_postproc:
                inp_rev = [self.gen_seq_postproc(seq) for seq in inp_rev]
                gen_rev = [self.gen_seq_postproc(seq) for seq in gen_rev]

            output_dc[OutDataF.GROUP_ID] += rev_group_id
            output_dc[OutDataF.INP_REV] += inp_rev
            output_dc[OutDataF.GEN_REV] += gen_rev
            output_dc[OutDataF.INP_REV_RATING] += rating
            output_dc[OutDataF.REV_INDX] += rev_indx

            # for analytics
            inp_seq_len = [_comp_seq_len(seq) for seq in inp_rev]
            gen_seq_len = [_comp_seq_len(seq) for seq in gen_rev]
            output_dc[OutDataF.INP_REV_LEN] += inp_seq_len
            output_dc[OutDataF.GEN_REV_LEN] += gen_seq_len

            #  props
            props = {n: batch[o].tolist() for o, n
                     in prop_old_new_fnames.items()}
            props = dct_list_to_list_dict(props)
            output_dc[OutDataF.PROPS] += props
            if len(pred_props):
                pred_props = dct_list_to_list_dict(pred_props)
                output_dc[OutDataF.PRED_PROPS] += pred_props

        # some models don't output predicted props this condition deals with it
        if not len(output_dc[OutDataF.PRED_PROPS]):
            del output_dc[OutDataF.PRED_PROPS]

        # running analytics of text
        if self.seq_analytics:
            for text_fname in [OutDataF.INP_REV, OutDataF.GEN_REV]:
                formatted_seqs = []
                for seq in output_dc[text_fname]:
                    seq = seq if not isinstance(seq, list) else " ".join(seq)
                    formatted_seqs.append(seq)
                fscores = format_stats(self.seq_analytics(formatted_seqs),
                                       f"`{text_fname.upper()}` Text Analytics")
                for s in fscores:
                    logger.info(s)

        output_dc.to_json(f=codecs.open(out_file_path, 'w', 'utf-8'),
                          grouping_fnames=[OutDataF.GROUP_ID, OutDataF.REV_INDX])
        logger.info("Generated sequences and saved to: '%s'." % out_file_path)

    def after_ep_wrapper(self, out_dir_path, checkpoint_fn=None,
                         summ_eval_data_source=None, summ_eval_kwargs=None):
        """Creates a function (decorator) that can be executed after each epoch.

        The decorator function takes as input only one (optional) argument:
        `epoch`.
        """
        summ_eval_kwargs = {} if summ_eval_kwargs is None else summ_eval_kwargs

        def after_ep_func(epoch=None):
            new_out_path = comb_paths(out_dir_path, "out_ep%d" % epoch) \
                if epoch else comb_paths(out_dir_path, "out")
            safe_mkdir(new_out_path)

            # saving the state
            if checkpoint_fn is not None and epoch is not None:
                new_checkpoint_fn = checkpoint_fn if epoch is None else \
                    "ep%d_%s" % (epoch, checkpoint_fn)
                out_fp = comb_paths(out_dir_path, new_checkpoint_fn)
                self.imodel.save_state(out_fp)

            # running evaluation against gold summaries
            if summ_eval_data_source is not None:
                self.summ_eval(new_out_path, summ_eval_data_source,
                               **summ_eval_kwargs)
        return after_ep_func

    def summ_gen_wrapper(self, batch, **kwargs):
        """Inputs a batch and returns a list of summaries (strings)."""
        if self.gen_seq_postproc is None:
            raise ValueError("Please provide `seq_postproc` to the constructor "
                             "of the class.")
        gen_summ, pred_props = self.imodel.generate(batch=batch, **kwargs)
        res = []
        for _gen_summ in gen_summ:
            _gen_summ = self.gen_seq_postproc(_gen_summ)
            if isinstance(_gen_summ, list):
                _gen_summ = " ".join(_gen_summ)
            res.append(_gen_summ)
        return res, pred_props

    def train(self, data_source, logging_steps=10, epoch=None, **kwargs):
        """Performs a single epoch training on the passed data_source.

        The method is adapted to multi-task learning where summaries are also
        used for training.

        Adjusted to reset accumulated statistics to `logging_steps`.

        :param data_source: self-explanatory.
        :param logging_steps: self-explanatory.
        """
        empty_cache()
        logger.info("Training data source: %s." % data_source)

        stat_coll = OrderedDict()
        size_coll = OrderedDict()

        du_count = 0
        dc_count = 0
        start = None
        for batch in self.train_data_pipeline.iter(**data_source):
            start = time() if start is None else start
            new_stats = self.imodel.train(batch=batch, **kwargs)
            batch_size = len(batch[ModelF.SUMM]) if ModelF.SUMM in batch \
                else len(batch[ModelF.REV])
            du_count += batch_size
            dc_count += 1

            # canceling batch averaging and accumulating statistics
            new_sizes = {k: batch_size for k in new_stats}
            new_stats = {k: v*batch_size for k, v in new_stats.items()}
            accum_stats(stat_coll, new_stats)
            accum_stats(size_coll, new_sizes)

            # logging and resetting the statistics
            if dc_count % logging_steps == 0:
                stat_coll = {k: v / size_coll[k] for k, v in stat_coll.items()}
                mess = stats_to_str(stat_coll, prefix="Chunk # %d" % dc_count)
                logger.info(mess)
                stat_coll = OrderedDict()
                size_coll = OrderedDict()

        end = time() - start
        logger.info(f"Total data-chunks: {dc_count} and data-units: {du_count}.")
        logger.info("Epoch training time elapsed: %.2f (s)." % end)
        logger.info("Data-unit/sec: %.3f." % (du_count / end))

    def eval(self, data_source, **kwargs):
        """
        Runs the model for each batch and collects/accumulates its internal
        stats (e.g.  loss, kld), which are assumed to be averaged over the
        number of data-units. Then aggregates by the total number of data-units
        division.

        The method is adapted to multi-task learning where summaries are also
        used for training.
        """
        empty_cache()
        logger.info("Evaluation data source: %s." % data_source)
        coll_stats = OrderedDict()
        coll_sizes = OrderedDict()
        du_count = 0
        dc_count = 0
        start = None
        for batch in self.val_data_pipeline.iter(**data_source):
            start = time() if start is None else start
            stats = self.imodel.eval(batch=batch, **kwargs)
            batch_size = len(batch[ModelF.SUMM]) if ModelF.SUMM in batch \
                else len(batch[ModelF.REV])

            for k, v in stats.items():
                if k not in coll_stats:
                    coll_stats[k] = 0.
                if k not in coll_sizes:
                    coll_sizes[k] = 0
                coll_stats[k] += v * batch_size  # rescaling back
                coll_sizes[k] += batch_size
            du_count += batch_size
            dc_count += 1

        end = time() - start
        logger.info(f"Total data-chunks: {dc_count} and data-units: {du_count}.")
        logger.info("Evaluation time elapsed: %.2f (s)." % (time() - start))
        logger.info("Data-unit/sec: %.3f." % (du_count / end))

        # compute the actual average over data-units
        rescaled_stats = OrderedDict()
        for k, v in coll_stats.items():
            rescaled_stats[k] = v / coll_sizes[k]

        return rescaled_stats


def _comp_seq_len(seq):
    """Assumes that the sequence is either list of sentences or list of words."""
    if isinstance(seq, list):
        new_seq = " ".join(seq)
    else:
        raise ValueError("Can't handle computation of sequence length.")
    seq_len = len(new_seq.split(" "))
    return seq_len


def _comp_rev_indx(group_id):
    dct = dict()
    res = []
    for _group_id in group_id:
        if _group_id not in dct:
            dct[_group_id] = 0
        dct[_group_id] += 1
        res.append(dct[_group_id])
    return res
