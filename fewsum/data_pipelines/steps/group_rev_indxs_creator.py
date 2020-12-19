from mltoolkit.mldp.steps.transformers import BaseTransformer
from fewsum.utils.helpers.data import group_vals_by_keys
from mltoolkit.mldp.utils.helpers.nlp.sequences import pad_sequences as pad_seqs
from fewsum.utils.fields import ModelF
import numpy as np


class GroupRevIndxsCreator(BaseTransformer):
    """Creates the necessary index fields to perform summarization of reviews.

    Creates special fields:
        1. GROUP_REV_INDXS: padded indices of data-units that belong
                           to the same group, and thus should be summarizer
                           all together.
        2. GROUP_REV_INDXS_MASK: contains binary value for masking dummy reviews
                                of groups that don't have the maximum number
                                of reviews.
        3. SUMM_PRODUCT_ID: self-explanatory.
        4. SUMM_CATEGORY: categories of groups that are summarized.

        Preserves the sequential order present in data-chunks.

    Makes data-chunks 'invalid' as summaries number will differ from the reviews
    number.
    """

    def __init__(self, rev_group_id_fname, rev_cat_fname, **kwargs):
        super(GroupRevIndxsCreator, self).__init__(**kwargs)
        self.rev_group_id_fname = rev_group_id_fname
        self.rev_cat_fname = rev_cat_fname

    def _transform(self, data_chunk):
        rev_group_ids = data_chunk[self.rev_group_id_fname]
        rev_cats = data_chunk[self.rev_cat_fname]

        group_map = group_vals_by_keys(range(len(rev_group_ids)), rev_group_ids)
        group_ids = list(group_map.keys())
        gr_indxs = list(group_map.values())

        # the gr_indxs are indxs reviews that belong to the same group
        padded_indxs, mask = pad_seqs(gr_indxs, pad_symbol=0,
                                      padding_mode='right')

        group_id_to_cat = group_vals_by_keys(rev_cats, rev_group_ids)
        cats = [group_id_to_cat[gr_id][0] for gr_id in group_ids]

        data_chunk[ModelF.CAT] = cats
        data_chunk[ModelF.GROUP_ID] = group_ids
        data_chunk[ModelF.GROUP_REV_INDXS] = padded_indxs
        data_chunk[ModelF.GROUP_REV_INDXS_MASK] = mask

        return data_chunk
