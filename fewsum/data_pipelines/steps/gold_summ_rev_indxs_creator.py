from mltoolkit.mldp.steps.transformers import BaseTransformer
from fewsum.utils.helpers.data import group_vals_by_keys
from mltoolkit.mldp.utils.helpers.nlp.sequences import pad_sequences as pad_seqs
from fewsum.utils.fields import ModelF


class GoldSummRevIndxsCreator(BaseTransformer):
    """
    The step is specific to data that has golden summaries, which are passed
    along the pipeline. E.g., Yelp or Amazon gold datasets.

    It will align golden summaries to reviews (by creating indxs), which come
    sorted, while the gold summaries retain their original order.

    I can't reuse a similar step (prod_rev_indxs_creator) because it produces
    summaries by scanning prod_ids of reviews, which come sorted. This step
    iterates over SUMM_GROUP_ID field and grabs reviews that belong to that group.

    If I do the same operation for indxs creation as I do when summaries are not
    available, then I might produce alignment mismatches. In order words,
    I would not be able to tell which summaries are first, which are second
    based on sorted prod_ids. Because golden summaries are not sorted!

    Creates special fields:
        1. GROUP_REV_INDXS: that contain indices (padded) of data units that
                            belong to the same group
        2. GROUP_REV_INDXS_MASK: padding mask for those data-units.
    """

    def __init__(self, **kwargs):
        super(GoldSummRevIndxsCreator, self).__init__(**kwargs)

    def _transform(self, data_chunk):
        rev_group_ids = data_chunk[ModelF.REV_GROUP_ID]
        group_ids = data_chunk[ModelF.GROUP_ID]

        groups = group_vals_by_keys(range(len(rev_group_ids)), rev_group_ids)

        aligned_rev_indxs = []
        for group_id in group_ids:
            aligned_rev_indxs.append(groups[group_id])

        # the gr_indxs are indxs reviews that belong to the same group
        padded_rev_indxs, mask = pad_seqs(aligned_rev_indxs, pad_symbol=0)

        data_chunk[ModelF.GROUP_REV_INDXS] = padded_rev_indxs
        data_chunk[ModelF.GROUP_REV_INDXS_MASK] = mask

        return data_chunk
