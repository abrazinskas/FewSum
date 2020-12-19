from mltoolkit.mlutils.helpers.formatting.general import stats_to_str


def dct_list_to_list_dict(dct):
    """Converts dictionary of lists to list of dictionaries."""
    res = [dict(zip(dct, i)) for i in zip(*dct.values())]
    return res


def format_stats(stats, title=None):
    res = []
    if title is not None:
        res.append("###  %s  ###" % title)
    for k, v in stats.items():
        if isinstance(v, dict):
            res.append("   - %s %s" % (k, stats_to_str(v)))
        else:
            res.append("   - %s: %.3f" % (k, v))
    return res
