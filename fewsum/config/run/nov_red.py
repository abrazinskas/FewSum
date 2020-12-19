from fewsum.config.run.base_run import BaseRunConfig
from fewsum.utils.helpers.registries import register_run_config
from mltoolkit.mlutils.tools import ExperimentsPathController
from fewsum.utils.constants import NOV_RED


@register_run_config(NOV_RED)
class NovRedRunConfig(BaseRunConfig):
    """Configuration for running the novelty reduction phase."""

    def __init__(self):
        super(NovRedRunConfig, self).__init__()
        self.exper_descr = "Novelty reduction phase."
        self.cuda_device_ids = [0]
        self.epochs = 18
        self.learning_rate = 6e-5

        #   GENERAL DATA RELATED  #
        # multiplication by the number of devices is needed in order to
        # make it work in the multi-gpu setup
        self.train_groups_per_batch = 5 * len(self.cuda_device_ids)
        self.val_groups_per_batch = 14 * len(self.cuda_device_ids)
        self.eval_groups_per_batch = 13 * len(self.cuda_device_ids)

        #   DATA SOURCES  #
        self.train_early_term = 13000
        self.val_early_term = 500

        #   GENERAL PATHS   #
        self.experiments_folder = 'nov_red'
        self.output_dir = 'runs/%s/%s' % (self.dataset, self.experiments_folder)
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)

        self.checkpoint_path = 'artifacts/amazon/checkpoints/novelty_reduction.tar'
