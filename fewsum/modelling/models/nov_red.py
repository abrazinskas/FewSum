from .basesum import BaseSum
from collections import OrderedDict
from mltoolkit.mlmo.utils.helpers.pytorch.computation import seq_log_prob, perpl_per_word
from fewsum.utils.helpers.computation import comp_cov_cmass
from fewsum.utils.helpers.registries import register_model
from fewsum.utils.constants import NOV_RED


@register_model(NOV_RED)
class NovRed(BaseSum):
    """Additional loss that forces the output word probabilities to be more
    aligned with words present in source/other reviews of the product.
    """

    def __init__(self, alpha=1., **kwargs):
        super(NovRed, self).__init__(**kwargs)
        self.alpha = alpha


    def forward(self, rev, rev_mask, other_rev_indxs,
                other_rev_indxs_mask,
                other_rev_comp_states, other_rev_comp_states_mask,
                other_rev_uwords, other_rev_uwords_mask,
                len_prop, rating_prop, rouge_prop, pov_prop):
        """
        Args:
            other_rev_uwords (LongTensor): unique word ids in for each review
                in `rev`.
                [batch_size, max_words]
            other_rev_uwords_mask (FloatTensor): the corresponding mask for padding.
                [batch_size, max_words]
        """

        rev_len = rev_mask.sum(-1)

        mem, \
        mem_bin_mask = self.create_mem(rev=rev, rev_mask=rev_mask,
                                       group_rev_indxs=other_rev_indxs,
                                       group_rev_indxs_mask=other_rev_indxs_mask,
                                       group_rev_comp_states=other_rev_comp_states,
                                       group_rev_comp_states_mask=other_rev_comp_states_mask)
        word_lprobs, \
        tr_state, mem_att_wts = self._decode(tgt=rev, mem=mem,
                                             mem_bin_mask=mem_bin_mask,
                                             len_prop=len_prop,
                                             rating_prop=rating_prop,
                                             rouge_prop=rouge_prop,
                                             pov_prop=pov_prop)
        # the first sub-loss
        nll = - seq_log_prob(word_lprobs[:, :-1], seq=rev[:, 1:],
                             seq_mask=rev_mask[:, 1:])
        ppl = perpl_per_word(nll=nll, lens=rev_len)
        avg_nll = nll.mean(0)
        avg_ppl = ppl.mean(0)

        # the second loss for coverage/novelty
        prob_cmass = comp_cov_cmass(word_lprobs, other_rev_uwords, other_rev_uwords_mask)
        summ_prob_cmass = (prob_cmass * rev_mask).sum(-1)

        cov_loss = (rev_len - summ_prob_cmass)
        avg_cov_loss = cov_loss.mean(0)

        avg_loss = avg_nll + self.alpha * avg_cov_loss

        #   STATISTICS   #

        stats = OrderedDict()
        stats['ppl'] = avg_ppl.item()
        stats['nll'] = avg_nll.item()
        stats['cov'] = avg_cov_loss.item()

        return avg_loss, stats
