
def accum_stats(old_stats, new_stats):
    """Accumulates statistics in-place by updating `old_stats`."""
    if not len(old_stats):
        for k in new_stats:
            old_stats[k] = new_stats[k]
    else:
        for k in new_stats.keys():
            assert k in old_stats
            old_stats[k] += new_stats[k]
