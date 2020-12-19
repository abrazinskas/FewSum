class OutDataF(object):
    """Output fields for saving the model's output to files. """

    CAT = 'category'
    GROUP_ID = 'group_id'

    REV_INDX = 'rev_indx'
    INP_REV = 'inp_rev'
    GEN_REV = 'gen_rev'
    GEN_SUMM = 'gen_summ'

    GOLD_SUMMS = 'gold_summs'

    # for summary eval
    ROUGE = 'rouge'

    # for extra analysis
    INP_REV_RATING = 'inp_rev_rating'
    INP_REV_LEN = 'inp_rev_len'
    GEN_REV_LEN = 'gen_rev_len'
    CONTENT_SUPPORT = 'cont_supp'

    # props
    PROPS = 'props'
    PRED_PROPS = 'pred_props'
    ROUGE_PROP = 'rouge'
    LEN_PROP = 'len'
    RATING_PROP = 'rating'
    POV_PROP = 'pov'
