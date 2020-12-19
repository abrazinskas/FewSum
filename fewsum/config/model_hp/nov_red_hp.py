from fewsum.utils.constants import NOV_RED
from fewsum.utils.helpers.registries import register_hp_config
from fewsum.config.model_hp import BaseSumHP


@register_hp_config(NOV_RED)
class NovRedHP(BaseSumHP):
    """Hyper-parameters for the novelty reduction phase."""

    def __init__(self):
        super(NovRedHP, self).__init__()
        self.alpha = 2.
