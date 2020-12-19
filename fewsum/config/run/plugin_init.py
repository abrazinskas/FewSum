from fewsum.config.run.base_run import BaseRunConfig
from fewsum.utils.helpers.registries import register_run_config
from mltoolkit.mlutils.tools import ExperimentsPathController
from fewsum.utils.constants import PLUGIN_INIT


@register_run_config(PLUGIN_INIT)
class PluginInitRunConfig(BaseRunConfig):
    """Configuration for running the plug-in initialization phase."""

    def __init__(self):
        super(PluginInitRunConfig, self).__init__()
        self.exper_descr = "Plug-in initialization phase on reviews."
        self.cuda_device_ids = [0]
        self.epochs = 13
        self.learning_rate = 1e-05

        #   GENERAL DATA RELATED  #
        # multiplication by the number of devices is needed in order to
        # make it work in the multi-gpu setup
        self.train_groups_per_batch = 20 * len(self.cuda_device_ids)
        self.val_groups_per_batch = 50 * len(self.cuda_device_ids)
        self.eval_groups_per_batch = 17 * len(self.cuda_device_ids)

        #   DATA SOURCES  #
        self.train_early_term = None
        self.val_early_term = None

        #   GENERAL PATHS   #
        self.experiments_folder = 'plugin_init'
        self.output_dir = 'runs/%s/%s' % (self.dataset, self.experiments_folder)
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)

        self.checkpoint_path = 'artifacts/amazon/checkpoints/plugin_init.tar'

        #   FREEZING AND UNFREEZING   #
        self.modules_to_unfreeze = ['plugin']
