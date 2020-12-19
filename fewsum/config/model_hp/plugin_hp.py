from fewsum.utils.constants import PLUGIN_INIT, PLUGIN_TUNING, JOINT_TUNING
from fewsum.utils.helpers.registries import register_hp_config
from fewsum.config.model_hp import BaseSumHP


@register_hp_config(PLUGIN_INIT)
@register_hp_config(PLUGIN_TUNING)
@register_hp_config(JOINT_TUNING)
class PluginHP(BaseSumHP):
    """Hyper-parameters for a trainable plug-in network."""

    def __init__(self):
        super(PluginHP, self).__init__()
        self.plugin_dim = 30
        self.plugin_ff_dim = 20
        self.plugin_dropout = 0.4
        self.plugin_mem_att_dropout = 0.15
        self.plugin_nheads = 3
        self.plugin_nlayers = 3

