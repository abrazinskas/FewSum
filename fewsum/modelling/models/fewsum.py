from fewsum.modelling.models import PluginNetwork
from mltoolkit.mlmo.utils.helpers.pytorch.computation import seq_log_prob, perpl_per_word
from collections import OrderedDict
from fewsum.utils.helpers.registries import register_model
from fewsum.utils.constants import JOINT_TUNING


@register_model(JOINT_TUNING)
class FewSum(PluginNetwork):
    """Wrapper class for fine-tuning the plugin network and model parts on
    gold summaries.
    """

    def __init__(self, **kwargs):
        super(FewSum, self).__init__(**kwargs)

    def forward(self, rev, rev_mask,
                summ, summ_mask, summ_group_indx,
                group_rev_indxs, group_rev_indxs_mask):
        summ_len = summ_mask.sum(-1)

        mem,\
        mem_bin_mask = self.create_mem(rev=rev, rev_mask=rev_mask,
                                       group_rev_indxs=group_rev_indxs,
                                       group_rev_indxs_mask=group_rev_indxs_mask)
        # selecting mem for each summary
        mem = mem[summ_group_indx]
        mem_bin_mask = mem_bin_mask[summ_group_indx]

        props_to_use, _ = self.plugin(mem, mem_bin_mask)

        word_lprobs, tr_state, mem_att_wts = self._decode(tgt=summ, mem=mem,
                                                          mem_bin_mask=mem_bin_mask,
                                                          **props_to_use)

        nll = - seq_log_prob(word_lprobs[:, :-1], seq=summ[:, 1:],
                             seq_mask=summ_mask[:, 1:])
        ppl = perpl_per_word(nll=nll, lens=summ_len.float())
        avg_nll = nll.mean(0)
        avg_ppl = ppl.mean(0)
        avg_loss = avg_nll

        #   STATISTICS   #

        stats = OrderedDict()
        stats['ppl'] = avg_ppl.item()
        stats['nll'] = avg_nll.item()

        return avg_loss, stats
