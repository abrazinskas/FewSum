# search/decoding related helper functions


def traverse_table(time_step, beam_indx, back_pointers, elems_table,
                   format_func=None):
    """
    Walks back to construct the full hypothesis by traversing the passed table.

    :param time_step: the last time-step of the best candidate
    :param beam_indx: the beam index of the best candidate in the last time-step
    :param back_pointers: array [steps]
    :param elems_table: array of elements to traverse [steps, elements_count],
                        e.g. vocabulary word ids
    :param format_func: a function to format elements
    :return: hypothesis list of the size 'time_step'
    """
    hyp = []
    for j in range(len(back_pointers[:time_step]) - 1, -1, -1):
        elem = elems_table[j + 1][beam_indx]
        elem = elem if format_func is None else format_func(elem)
        hyp.append(elem)
        beam_indx = back_pointers[j][beam_indx]
    elem = elems_table[0][beam_indx]
    elem = elem if format_func is None else format_func(elem)
    hyp.append(elem)
    return hyp[::-1]


def adjust_tensors_to_beam_size(*tens, beam_size):
    res = (adjust_tensor_to_beam_size(t, beam_size=beam_size) for t in tens)
    return res


def adjust_tensor_to_beam_size(tens, beam_size):
    """Replicates tensor values for each beam over the first dim."""
    bs = tens.size(0)
    if len(tens.shape) == 3:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1, 1))
        tens = tens.view(bs * beam_size, s, -1)
    elif len(tens.shape) == 2:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1))
        tens = tens.view(bs * beam_size, s)
    elif len(tens.shape) == 1:
        tens = tens.unsqueeze(1).repeat((1, beam_size))
        tens = tens.view(bs * beam_size)
    else:
        raise NotImplementedError
    return tens
