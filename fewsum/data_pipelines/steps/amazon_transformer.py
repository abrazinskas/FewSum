from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.tools import DataChunk
from fewsum.utils.fields import GoldDataF, ModelF


class AmazonTransformer(BaseTransformer):
    """
    1. Collapses summary1, summary2, and summary3 entries to one one list.
    2. Splits review columns, such that one entry has one review.
    3. Copies some fields directly (e.g., category and product id, those will be
        associated with summaries or groups)
    """

    def __init__(self, fnames_to_copy):
        super(AmazonTransformer, self).__init__()
        if not isinstance(fnames_to_copy, list):
            fnames_to_copy = [fnames_to_copy]
        self.fnames_to_copy = fnames_to_copy

    def _transform(self, data_chunk):
        new_dc = DataChunk()
        new_dc[ModelF.SUMMS] = []

        for summ1, summ2, summ3 in zip(data_chunk[GoldDataF.SUMM1],
                                       data_chunk[GoldDataF.SUMM2],
                                       data_chunk[GoldDataF.SUMM3]):
            _summs = []
            if isinstance(summ1, str):
                _summs.append(summ1)
            if isinstance(summ2, str):
                _summs.append(summ2)
            if isinstance(summ3, str):
                _summs.append(summ3)
            new_dc[ModelF.SUMMS].append(_summs)

        # copy some fields directly
        for fname in self.fnames_to_copy:
            new_dc[fname] = data_chunk[fname]
            new_dc[fname] = data_chunk[fname]

        # splitting data-units by the reviews field. I.e. each unit will have
        # one review associated with it
        new_to_old = {ModelF.REV_GROUP_ID: GoldDataF.PROD_ID,
                      ModelF.REV_CAT: GoldDataF.CAT}

        for new_fname in new_to_old.keys():
            new_dc[new_fname] = []
        new_dc[ModelF.REV] = []
        for du in data_chunk.iter():
            for rev_fn in GoldDataF.REVS:
                new_dc[ModelF.REV].append(du[rev_fn])
                # copying the rest
                for new_fn, old_fn in new_to_old.items():
                    new_dc[new_fn].append(du[old_fn])
        return new_dc
