from .base_interface import BaseInterface
from .i_base_model import IBaseModel
from mltoolkit.mldp import Pipeline
from logging import getLogger
from mltoolkit.mlutils.helpers.formatting.general import format_big_box, stats_to_str
from mltoolkit.mlmo.utils.helpers.general import accum_stats
from collections import OrderedDict
import os
import warnings
from time import time


logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class IBaseDev(BaseInterface):
    """
    Base class for modelling development interfaces. Interfaces of this type
    contain logic specific for development purposes, such as how to eval
    the model.
    """
    
    def __init__(self, imodel, train_data_pipeline, val_data_pipeline=None,
                 name_prefix=None):
        if not isinstance(imodel, IBaseModel):
            raise TypeError("Please provide a valid interface of a model.")
        if not isinstance(train_data_pipeline, Pipeline):
            raise ValueError("Please provide a valid training data pipeline.")
        if not isinstance(val_data_pipeline, Pipeline):
            raise ValueError("Please provide a valid validation data pipeline.")
        super(IBaseDev, self).__init__(name_prefix=name_prefix)
        self.imodel = imodel
        self.train_data_pipeline = train_data_pipeline
        self.val_data_pipeline = val_data_pipeline if val_data_pipeline else\
            train_data_pipeline

    def save_setup_str(self, dir_path, exper_descr=None):
        """
        Logs/saves the setup of the dev. experiment, namely 3 main components:
        1. Experiment's description -> experiment.txt (if provided exper_descr)
        2. dev. data pipeline's blueprint -> dp.txt
        3. model's blueprint/summary -> model.txt
        """
        logger.info("Experiment's output will be saved to: '%s'." % dir_path)
        # 1. experiment
        if exper_descr:
            form_exp = format_big_box(exper_descr)
            logger.info(form_exp)

        # 2. data pipeline
        dp_fp = os.path.join(dir_path, 'dp.txt')
        try:
            with open(dp_fp, 'w') as f:
                f.write(str(self.train_data_pipeline))
        except Exception:
            os.remove(dp_fp)
            warnings.warn("Could not get the str of the train_data_pipeline's setup.")

        # 3. model and its interface
        m_fp = os.path.join(dir_path, 'model.txt')
        try:
            with open(m_fp, 'w') as f:
                f.write(str(self.imodel))
        except Exception:
            os.remove(m_fp)
            warnings.warn("Could not get the str of the model's setup.")

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
        if val_data_source:
            stats = self.eval(data_source=val_data_source)
            logger.info(stats_to_str(stats, "Validation"))

        epoch = 0
        for epoch in range(1, epochs + 1):
            logger.info('Epoch %d/%d' % (epoch, epochs))
            self.train(data_source=train_data_source, epoch=epoch,
                       logging_steps=logging_period)

            if eval_train:
                stats = self.eval(data_source=train_data_source,
                                  epoch=epoch)
                if stats:
                    logger.info(stats_to_str(stats, "Training"))

            if val_data_source:
                stats = self.eval(data_source=val_data_source)
                logger.info(stats_to_str(stats, "Validation"))

            if after_epoch_func:
                after_epoch_func(epoch)

        if test_data_source:
            stats = self.eval(data_source=test_data_source,
                              epoch=epoch)
            logger.info(stats_to_str(stats, "Testing"))

    def train(self, data_source, logging_steps=10, epoch=None, **kwargs):
        """
        Performs a single epoch training on the passed data_source.

        :param data_source: self-explanatory.
        :param logging_steps: self-explanatory.
        """
        logger.info("Training data source: %s." % data_source)
        du_count = 0
        dc_count = 0
        stat_coll = OrderedDict()
        size_coll = OrderedDict()

        start = time()
        for i, batch in enumerate(self.train_data_pipeline.iter(**data_source), 1):
            new_stats = self.imodel.train(batch=batch, **kwargs)

            batch_size = len(batch)
            du_count += batch_size
            dc_count += 1

            # canceling batch averaging and accumulating statistics
            new_stats = {k: v*batch_size for k, v in new_stats.items()}
            new_sizes = {k: batch_size for k in new_stats}
            accum_stats(stat_coll, new_stats)
            accum_stats(size_coll, new_sizes)

            if i % logging_steps == 0:
                stat_coll = {k: v / size_coll[k] for k, v in stat_coll.items()}
                mess = stats_to_str(stat_coll, prefix="Chunk # %d" % i)
                logger.info(mess)
                stat_coll = OrderedDict()
                size_coll = OrderedDict()

        end = time() - start
        logger.info("Total data-units: %d." % du_count)
        logger.info("Total data-chunks: %d." % dc_count)
        logger.info("Epoch training time elapsed: %.2f (s)." % end)
        logger.info("Units/sec: %.3f." % (du_count / end))

    def eval(self, data_source, **kwargs):
        """
        Runs the model for each batch and collects/accumulates its internal
        stats (e.g.  loss, kld), which are assumed to be averaged over the
        number of data-units. Then aggregates by the total number of data-units
        division.
        """
        logger.info("Evaluation data source: %s." % data_source)
        total_stats = OrderedDict()
        total_dus = 0
        start = time()

        for batch in self.val_data_pipeline.iter(**data_source):
            stats = self.imodel.eval(batch=batch, **kwargs)
            for k, v in stats.items():
                if k not in total_stats:
                    total_stats[k] = 0.
                total_stats[k] += v * len(batch)  # rescaling back
            total_dus += len(batch)

        logger.info("Total data-units: %d." % total_dus)
        logger.info("Evaluation time elapsed: %.2f (s)." % (time() - start))

        # compute the actual average over data-units
        rescaled_stats = OrderedDict()
        for k, v in total_stats.items():
            rescaled_stats[k] = v / float(total_dus)

        return rescaled_stats
