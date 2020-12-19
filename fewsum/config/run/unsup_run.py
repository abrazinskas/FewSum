from fewsum.config.run.base_run import BaseRunConfig
from fewsum.utils.helpers.registries import register_run_config
from mltoolkit.mlutils.tools import ExperimentsPathController
from fewsum.utils.constants import UNSUP


@register_run_config(UNSUP)
class UnsupRunConfig(BaseRunConfig):
    """Configuration for running the unsupervised model."""

    def __init__(self):
        super(UnsupRunConfig, self).__init__()
        self.exper_descr = "Unsupervised learning phase with the leave-one-out " \
                           "objective and Oracle"
        self.cuda_device_ids = [0]
        self.epochs = 20
        self.learning_rate = 6e-05

        #   GENERAL DATA RELATED  #
        # multiplication by the number of devices is needed in order to
        # make it work in the multi-gpu setup
        self.train_groups_per_batch = 6 * len(self.cuda_device_ids)
        self.val_groups_per_batch = 20 * len(self.cuda_device_ids)
        self.eval_groups_per_batch = 13 * len(self.cuda_device_ids)

        #   DATA SOURCES  #
        self.train_early_term = 5000
        self.val_early_term = 500

        #   GENERAL PATHS   #
        self.experiments_folder = 'unsup'
        self.output_dir = 'runs/%s/%s' % (self.dataset, self.experiments_folder)
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)

        self.checkpoint_path = 'artifacts/amazon/checkpoints/unsupervised.tar'
