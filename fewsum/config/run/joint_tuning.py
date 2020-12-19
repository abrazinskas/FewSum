from fewsum.config.run.base_run import BaseRunConfig
from fewsum.utils.helpers.registries import register_run_config
from mltoolkit.mlutils.tools import ExperimentsPathController
from fewsum.utils.constants import JOINT_TUNING


@register_run_config(JOINT_TUNING)
class JointTuningRunConfig(BaseRunConfig):
    """Configuration for running the final tuning phase. Both the plug-in network
    and the memory attention module.

    At the moment, can be run only on a single GPU.
    """

    def __init__(self):
        super(JointTuningRunConfig, self).__init__()
        self.exper_descr = "Joint tuning: the plug-in network and the memory attention."
        self.epochs = 33
        self.learning_rate = 0.0001

        #   GENERAL DATA RELATED  #
        self.train_groups_per_batch = 15
        self.val_groups_per_batch = 20
        self.eval_groups_per_batch = 17

        #   DATA SOURCES  #
        self.train_early_term = None
        self.val_early_term = None

        #   GENERAL PATHS   #
        self.experiments_folder = 'joint_tuning'
        self.output_dir = 'runs/%s/%s' % (self.dataset, self.experiments_folder)
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)

        self.checkpoint_path = 'artifacts/amazon/checkpoints/joint_tuning.tar'
        # self.checkpoint_path = None

        #   FREEZING AND UNFREEZING   #
        self.modules_to_unfreeze = ['plugin',
                                    '_tr_stack.layers.0.mem_attn',
                                    '_tr_stack.layers.1.mem_attn',
                                    '_tr_stack.layers.2.mem_attn',
                                    '_tr_stack.layers.3.mem_attn',
                                    '_tr_stack.layers.4.mem_attn',
                                    '_tr_stack.layers.5.mem_attn']
