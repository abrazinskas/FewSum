import torch as T
from mltoolkit.mlmo.utils.helpers.pytorch.computation import seq_log_prob, perpl_per_word
from torch.nn import Module, Sequential, Linear, ReLU
from torch.nn.functional import log_softmax
from collections import OrderedDict
from mltoolkit.mlmo.utils.tools import DecState
from fewsum.modelling.modules import TransformerStack, TransformerEmbeddings
from fewsum.utils.helpers.modelling import group_att_over_input
from fewsum.modelling.modules.transformer_stack import MEM_ATT_WTS
from fewsum.utils.helpers.registries import register_model
from fewsum.utils.constants import UNSUP


EPS = 1e-8


@register_model(UNSUP)
class BaseSum(Module):
    """Base version of the summarizer. Used for unsupervised pre-training.

    Other models extends this class as the encoder-generator architecture remains
    the same.
    """

    def __init__(self, word_emb_num, word_emb_dim, len_emb_dim, word_emb_dropout,
                 ff_dim, dropout, mem_att_dropout, nheads, nlayers,
                 len_emb_num=200, drop_enc_embds=False):
        super(BaseSum, self).__init__()

        len_emb_num = len_emb_num + 2  # + 2 because of wrapper symbols
        self.word_emb_dim = word_emb_dim
        self.drop_enc_embds = drop_enc_embds

        # length embeddings are concatenated to input word embeddings
        self.model_dim = word_emb_dim + len_emb_dim

        self._embds = TransformerEmbeddings(word_emb_num, word_emb_dim,
                                            dropout=word_emb_dropout,
                                            len_embd_dim=len_emb_dim,
                                            max_len=len_emb_num)
        #   Transformer   #
        prop_dim = 9  # len + polarity + 4 pov + 3 ROUGE
        self._tr_stack = TransformerStack(model_dim=self.model_dim,
                                          num_layers=nlayers, dropout=dropout,
                                          mem_att_dropout=mem_att_dropout,
                                          ff_dim=ff_dim, prop_dim=prop_dim,
                                          nheads=nheads, mem_att=True)
        #   - word distribution function   #
        self._dec_ffnn = Sequential()
        self._dec_ffnn.add_module("lin", Linear(self.model_dim, word_emb_dim))
        self._dec_ffnn.add_module("emb", self._embds)

    def forward(self, rev, rev_mask, other_rev_indxs,
                other_rev_indxs_mask,
                other_rev_comp_states, other_rev_comp_states_mask,
                len_prop, rating_prop, rouge_prop, pov_prop):
        """
        Args:
            rev (LongTensor): customer reviews (word ids)
                [batch_size, seq_len]
            rev_mask (FloatTensor): binary mask for review word ids.
                [batch_size, seq_len]
            other_rev_indxs (LongTensor): indexes of 'other' reviews for a each
                left out review in `rev`.
                [batch_size, max_group_size]
            other_rev_indxs_mask (FloatTensor): mask for indexes.
            other_rev_comp_states (LongTensor): indexes of non-masked states of
                encoded 'other' reviews. Used for computational efficiency.
            other_rev_comp_states_mask (FloatTensor): mask for padded review
                encoded states.
            len_prop: length deviation of each review in `rev` from the other ones.
                [batch_size, 1]
            rating_prop: rating deviation of each review in `rev` from the other ones.
                [batch_size, 1]
            rouge_prop: coverage property for each review in `rev`.
                [batch_size, 3]
            pov_prop: points-of-view property for each review in `rev`.
                [batch_size, 4]
        """
        rev_len = rev_mask.sum(-1)

        mem, mem_bin_mask = self.create_mem(rev=rev, rev_mask=rev_mask,
                                            group_rev_indxs=other_rev_indxs,
                                            group_rev_indxs_mask=other_rev_indxs_mask,
                                            group_rev_comp_states=other_rev_comp_states,
                                            group_rev_comp_states_mask=other_rev_comp_states_mask)

        word_lprobs,\
        tr_state, mem_att_wts = self._decode(tgt=rev, mem=mem,
                                             mem_bin_mask=mem_bin_mask,
                                             len_prop=len_prop,
                                             rating_prop=rating_prop,
                                             rouge_prop=rouge_prop,
                                             pov_prop=pov_prop)

        nll = - seq_log_prob(word_lprobs[:, :-1], seq=rev[:, 1:],
                             seq_mask=rev_mask[:, 1:])
        ppl = perpl_per_word(nll=nll, lens=rev_len)
        avg_nll = nll.mean(0)
        avg_ppl = ppl.mean(0)
        avg_loss = avg_nll

        #   STATISTICS   #

        stats = OrderedDict()
        stats['ppl'] = avg_ppl.item()
        stats['nll'] = avg_nll.item()

        return avg_loss, stats

    def _encode(self, src, src_bin_mask):
        emb = self._embds(src, dropout=self.drop_enc_embds)
        out, tr_states, tr_arts = self._tr_stack(tgt=emb, decode=False,
                                                 tgt_key_padding_mask=src_bin_mask)
        return out

    def _decode(self, tgt, mem=None, mem_bin_mask=None,
                tr_state=None, pos_offset=0, log_normalize=True, **props):
        word_embd = self._embds(tgt, pos_offset=pos_offset)
        props = _prepare_props(**props).unsqueeze(1).repeat(1, word_embd.size(1), 1)
        out, tr_state, tr_arts = self._tr_stack(tgt=word_embd, states=tr_state,
                                                props=props, mem=mem,
                                                mem_key_padding_mask=mem_bin_mask,
                                                decode=True)
        mem_att_wts = tr_arts[MEM_ATT_WTS]
        scores = self._dec_ffnn(out)
        if log_normalize:
            word_log_probs = log_softmax(scores, dim=-1)
            return word_log_probs, tr_state, mem_att_wts
        else:
            return scores, tr_state, mem_att_wts

    def decode(self, seq, tr_state=None, dummy=None, **kwargs):
        """BeamSearch or Sampler specific decoding function. Performs one step
        decoding based on the current state in `tr_state`.

        Args:
            seq: [batch_size, 1]
            tr_state: [batch_size, num_layers, curr_seq_len, model_dim]
            dummy: it used for consistency with the beam search.

        Returns:
            DecState
        """
        pos_offset = 0 if tr_state is None else tr_state.size(2)
        tr_state = tr_state.transpose(1, 0) if tr_state is not None else None
        word_scores, new_tr_state, \
        mem_att_wts = self._decode(tgt=seq, tr_state=tr_state,
                                   pos_offset=pos_offset, **kwargs)
        # new_state: [num_layers, batch_size, curr_seq_len, model_dim]
        tr_state = new_tr_state if tr_state is None else\
            T.cat((tr_state, new_tr_state), dim=2)
        tr_state = tr_state.transpose(1, 0)
        out = DecState(word_scores=word_scores,
                       rec_vals={"tr_state": tr_state})
        return out

    def create_mem(self, rev, rev_mask, group_rev_indxs,
                   group_rev_indxs_mask, group_rev_comp_states=None,
                   group_rev_comp_states_mask=None):
        """Creates values and mask to attend over group review states.

        Returns:
            att_vals: [batch_size, att_seq_len, emb_dim]
            att_bin_mask: [batch_size, att_seq_len]
        """
        # index adjustment for multi-gpu setup
        group_rev_indxs -= T.min(group_rev_indxs)

        group_count = group_rev_indxs.size(0)
        device = rev.device
        rev_bin_mask = rev_mask == 0.
        rev_states = self._encode(rev, rev_bin_mask)

        att_vals, \
        att_mask = group_att_over_input(inp_att_vals=rev_states,
                                        inp_att_mask=rev_mask,
                                        att_indxs=group_rev_indxs,
                                        att_indxs_mask=group_rev_indxs_mask)

        if group_rev_comp_states is not None and group_rev_comp_states_mask \
                is not None:
            # optimizing the attention targets by making more compact tensors
            # with less padded entries
            sel = T.arange(group_count, device=device).unsqueeze(-1)
            att_vals = att_vals[sel, group_rev_comp_states]
            att_mask = group_rev_comp_states_mask

        att_bin_mask = att_mask == 0.

        return att_vals, att_bin_mask


def _prepare_props(len_prop, rating_prop, rouge_prop, pov_prop):
    """Formats properties to a tensor to be fed to the model."""
    props = T.cat([len_prop.unsqueeze(1), rating_prop.unsqueeze(1),
                   rouge_prop, pov_prop], dim=-1)
    return props
