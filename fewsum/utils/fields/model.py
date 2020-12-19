class ModelF(object):
    """Model related fields."""

    # interfaces
    GROUP_ID = 'group_id'
    CAT = 'cat'

    # core model
    REV_GROUP_ID = 'rev_group_id'
    REV_CAT = 'rev_cat'
    REV = 'rev'
    REV_LEN = 'rev_len'
    REV_MASK = 'rev_mask'
    REV_RATING = 'rev_rating'

    GROUP_REV_INDXS = 'group_rev_indxs'
    GROUP_REV_INDXS_MASK = 'group_rev_indxs_mask'

    # SUMM_GROUP_ID = 'summ_group_id'
    SUMMS = 'summs' # used for eval
    SUMM = 'summ'  # used for training/fine-tuning
    SUMM_LEN = 'summ_len'
    SUMM_MASK = 'summ_mask'
    SUMM_GROUP_INDX = 'summ_group_indx'
    # SUMM_CAT = 'summ_category'

    # attention
    REV_TO_GROUP_INDX = 'rev_to_group_indx'
    OTHER_REV_INDXS = 'other_rev_indxs'
    OTHER_REV_INDXS_MASK = 'other_rev_indxs_mask'
    OTHER_REV_COMP_STATES = 'other_rev_comp_states'
    OTHER_REV_COMP_STATES_MASK = 'other_rev_comp_states_mask'

    # extra loss
    OTHER_REV_UWORDS = 'other_rev_uwords' # unique words in other reviews
    OTHER_REV_UWORDS_MASK = 'other_rev_uwords_mask'

    # PROPERTIES
    LEN_PROP = 'len_prop'
    RATING_PROP = 'rating_prop'
    ROUGE_PROP = 'rouge_prop'
    POV_PROP = 'pov_prop'
