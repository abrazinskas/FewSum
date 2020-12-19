from mltoolkit.mlmo.utils.tools import BaseConfig
from fewsum.utils.constants import UNSUP
from fewsum.utils.helpers.registries import register_hp_config


@register_hp_config(UNSUP)
class BaseSumHP(BaseConfig):
    """The base hyper-parameters of the model."""

    def __init__(self):
        super(BaseSumHP, self).__init__()

        #   GENERAL   #
        self.word_emb_dim = 390
        self.len_emb_num = 110
        self.len_emb_dim = 10
        self.word_emb_dropout = 0.1

        #   TRANSFORMER (ENCODER-DECODER)   #
        self.drop_enc_embds = True
        self.ff_dim = 1000
        self.dropout = 0.10
        self.mem_att_dropout = 0.1
        self.nheads = 8
        self.nlayers = 6
