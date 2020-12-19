from fewsum.config.run.base_run import BaseRunConfig
from fewsum.utils.helpers.registries import register_run_config
from mltoolkit.mlutils.tools import ExperimentsPathController
from fewsum.utils.constants import PLUGIN_TUNING


@register_run_config(PLUGIN_TUNING)
class PluginTuningRunConfig(BaseRunConfig):
    """Configuration for running the plug-in tuning phase (on summaries).

    At the moment, can be run only on a single GPU.
    """

    def __init__(self):
        super(PluginTuningRunConfig, self).__init__()
        self.exper_descr = "Plug-in tuning phase on summaries."
        self.epochs = 98
        self.learning_rate = 0.0007

        #   GENERAL DATA RELATED  #
        self.train_groups_per_batch = 30
        self.val_groups_per_batch = 50
        self.eval_groups_per_batch = 17

        #   DATA SOURCES  #
        self.train_early_term = None
        self.val_early_term = None

        #   GENERAL PATHS   #
        self.experiments_folder = 'plugin_tuning'
        self.output_dir = 'runs/%s/%s' % (self.dataset, self.experiments_folder)
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)

        self.checkpoint_path = 'artifacts/amazon/checkpoints/plugin_tuning.tar'

        #   FREEZING AND UNFREEZING   #
        self.modules_to_unfreeze = ['plugin']
