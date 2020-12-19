from fewsum.modelling.models.basesum import BaseSum
from fewsum.utils.fields import ModelF
from mltoolkit.mlmo.utils.helpers.pytorch.computation import kld_cat
from fewsum.modelling.modules import Plugin
from fewsum.utils.helpers.registries import register_model
from fewsum.utils.constants import PLUGIN_INIT, PLUGIN_TUNING

EPS = 1e-12


@register_model(PLUGIN_INIT)
@register_model(PLUGIN_TUNING)
class PluginNetwork(BaseSum):
    """A wrapper to train/fine-tune the plugin network on true property values.

    Adapted to work both for unannotated and annotated data.
    """

    def __init__(self, plugin_dim=None, plugin_ff_dim=None, plugin_nheads=None,
                 plugin_nlayers=None, plugin_dropout=0.,
                 plugin_mem_att_dropout=0.,
                 len_scalar=0.1, rating_scalar=1.,
                 pov_scalar=0.08, rouge_scalar=0.5, **kwargs):
        super(PluginNetwork, self).__init__(
                                     **kwargs)
        self.scalars = {ModelF.LEN_PROP: len_scalar,
                        ModelF.RATING_PROP: rating_scalar,
                        ModelF.POV_PROP: pov_scalar,
                        ModelF.ROUGE_PROP: rouge_scalar}

        self.plugin = Plugin(model_dim=plugin_dim, ff_dim=plugin_ff_dim,
                             nheads=plugin_nheads, nlayers=plugin_nlayers,
                             mem_att_dropout=plugin_mem_att_dropout,
                             dropout=plugin_dropout, len_prop=True,
                             rouge_prop=True, pov_prop=True,
                             rating_prop=True,
                             mem_dim=self.model_dim)

    def forward(self, rev, rev_mask,
                len_prop, rating_prop, rouge_prop, pov_prop,
                # batch with unannotated data
                other_rev_indxs=None, other_rev_indxs_mask=None,
                other_rev_comp_states=None, other_rev_comp_states_mask=None,
                # batch with annotated data
                summ_group_indx=None, group_rev_indxs=None,
                group_rev_indxs_mask=None):
        assert (other_rev_indxs is not None and other_rev_indxs_mask is not None) or\
               (group_rev_indxs is not None and group_rev_indxs_mask is not None)

        indxs = other_rev_indxs if other_rev_indxs is not None \
            else group_rev_indxs
        indxs_mask = other_rev_indxs_mask if other_rev_indxs is not None \
            else group_rev_indxs_mask

        mem, \
        mem_bin_mask = self.create_mem(rev=rev, rev_mask=rev_mask,
                                       group_rev_indxs=indxs,
                                       group_rev_indxs_mask=indxs_mask,
                                       group_rev_comp_states=other_rev_comp_states,
                                       group_rev_comp_states_mask=other_rev_comp_states_mask)

        pred_knobs, mem_att_wts = self.plugin(mem, mem_bin_mask)

        if summ_group_indx is not None:
            pred_knobs = {k: v[summ_group_indx] for k, v in pred_knobs.items()}

        true_knobs = {ModelF.LEN_PROP: len_prop, ModelF.RATING_PROP: rating_prop,
                      ModelF.ROUGE_PROP: rouge_prop, ModelF.POV_PROP: pov_prop}
        losses = self._compute_prop_loss(pred_knobs=pred_knobs,
                                         true_knobs=true_knobs)

        losses = {k: (v * self.scalars[k]).mean(0) for k, v in losses.items()}

        stats = dict()
        avg_loss = sum(losses.values())
        stats['loss'] = avg_loss.item()

        for k, v in losses.items():
            stats[k] = v.item()

        return avg_loss, stats

    def _compute_prop_loss(self, pred_knobs, true_knobs):
        """
        Computes the plugin property value prediction loss loss:
            - length, rating, ROUGE : squared error.
            - POV : Kullback-Leibler [pred, true].
        """
        losses = {}
        pred_len = pred_knobs[ModelF.LEN_PROP]
        true_len = true_knobs[ModelF.LEN_PROP]
        losses[ModelF.LEN_PROP] = abs(pred_len - true_len)
        pred_rating = pred_knobs[ModelF.RATING_PROP]
        true_rating = true_knobs[ModelF.RATING_PROP]
        losses[ModelF.RATING_PROP] = abs(pred_rating - true_rating)
        pred_rouge = pred_knobs[ModelF.ROUGE_PROP]
        true_rouge = true_knobs[ModelF.ROUGE_PROP]
        losses[ModelF.ROUGE_PROP] = (abs(pred_rouge - true_rouge)).sum(-1)
        pred_pov = pred_knobs[ModelF.POV_PROP]
        true_pov = true_knobs[ModelF.POV_PROP]
        losses[ModelF.POV_PROP] = kld_cat(pred_pov, true_pov, eps=EPS)
        return losses
