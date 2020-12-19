def find_mirror_next(seq, max_window_size, mirror_centre):
    """Find the next token that will lead to a mirror pattern.

    Searches in the range of `window_size` tokens to the left from the
    `mirror_centre` if it's found.

    E.g., in [a, b, AND, a] the next token should be 'b' to create a mirror
    pattern.

    Args:
        seq (list): list of tokens or ids
        max_window_size (int): maximum span of search from the found
            `mirror_centre`.
        mirror_centre (list): list of tokens/ids that should be searched for as
            centres of mirror patterns.

    Returns:
        next unit that that will lead to a mirror pattern.
        if no units in the `max_window_size` are in `mirror_centre`, then it
        will return None.
    """
    assert max_window_size > 0

    for i in range(1, max_window_size + 1):
        if len(seq) < 2*i:
            continue
        centre_indx = len(seq) - i
        if seq[centre_indx] in mirror_centre:
            left = seq[centre_indx - i:centre_indx-1]
            right = seq[centre_indx + 1:]
            next_unit = seq[centre_indx-1]
            if left == right:
                return next_unit
