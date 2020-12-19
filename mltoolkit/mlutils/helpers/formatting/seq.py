import numpy as np
import torch as T


def conv_seqs_to_sents(seqs, excl_tokens=None, end_token='.'):
    """
    Converts sequences of tokens into a list of sentences (strings). Does not use
    a proper de-tokenizer, but instead just joins the tokens.

    :param seqs: list of lists (tokens).
    :param excl_tokens: tokens which should be excluded from the final
                        result.
    :return: array of lists, where each element of the list is a sentence string.
    """
    excl_tokens = excl_tokens if excl_tokens else {}
    res = np.empty(len(seqs), dtype='object')
    for indx, seq in enumerate(seqs):
        curr_seq_sents = []
        curr_sent = []
        for token in seq:
            if token in excl_tokens:
                continue
            curr_sent.append(token)
            if token == end_token:
                curr_seq_sents.append(" ".join(curr_sent))
                curr_sent = []
        if curr_sent:
            curr_seq_sents.append(" ".join(curr_sent))
        res[indx] = curr_seq_sents
    return res


def conv_seq_to_sent_symbols(seq, excl_symbols=None, end_symbol='.',
                             remove_end_symbol=True):
    """
    Converts sequences of tokens/ids into a list of sentences (tokens/ids).

    :param seq: list of tokens/ids.
    :param excl_symbols: tokens/ids which should be excluded from the final
                         result.
    :param end_symbol: self-explanatory.
    :param remove_end_symbol: whether to remove from each sentence the end
                              symbol.
    :return: list of lists, where each sub-list contains tokens/ids.
    """
    excl_symbols = excl_symbols if excl_symbols else {}
    assert end_symbol not in excl_symbols
    coll = []
    curr_sent = []
    for symbol in seq:
        if symbol in excl_symbols:
            continue
        if symbol == end_symbol:
            if not remove_end_symbol:
                curr_sent.append(symbol)
            coll.append(curr_sent)
            curr_sent = []
        else:
            curr_sent.append(symbol)
    if curr_sent:
        coll.append(curr_sent)
    return coll


def format_seq(seq, start, end, pad, retain_end=True):
    """Makes each sequence end when the first `end` or `pad` is encountered."""
    formatted_seq = []
    for id in seq:
        if id == start:
            continue
        if id == end:
            if retain_end:
                formatted_seq.append(id)
            break
        if id == pad:
            break
        formatted_seq.append(id)
    return formatted_seq
