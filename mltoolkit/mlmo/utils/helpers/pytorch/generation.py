import torch as T


def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    """Filters a distribution of logits using top-k filtering.

    Args:
        logits: predicted word logits.
            [batch_size, *, vocab_size]
        top_k >0: keep only top k tokens with highest probability.
            (top-k filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the
        # top-k
        indices_to_remove = logits < T.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    return logits
