import torch as T


def comp_cov_cmass(log_probs, words, words_mask):
    """Computes cumulative probability mass that is assigned to `words`.

    Args:
        log_probs: [batch_size, seq_len, vocab_size]
        words: [batch_size, word_count]
        words_mask: [batch_size, word_count]

    Return:
        prob_cmass: [batch_size, seq_len]
    """
    sl = log_probs.size(1)
    sel_log_probs = T.gather(input=log_probs, dim=-1,
                             index=words.unsqueeze(1).repeat(1, sl, 1))
    sel_probs = T.exp(sel_log_probs) * words_mask.unsqueeze(1)
    prob_cmass = sel_probs.sum(-1)
    return prob_cmass


def compute_pov_distr(tokens):
    """Computes a distribution over 3 points-of-view and one extra slot (other).

    Computation is based on pronouns, and the last class is assigned 100% of
    mass if no pronouns are present.

    Args:
        tokens (list): list of text tokens. 

    Returns:
        distr (list): 4 class distribution.
    """
    POVS = [
        {"I", "me", "myself", "my", "mine", "we", "us", "ourselves", "our",
         "ours"},
        {"you", "yourself", "your", "yours"},
        {"he", "she", "it", "him", "her", "his", "hers", "its", "they", "them",
         "their", "theirs"}
    ]
    counts = [0, 0, 0, 0]
    for tok in tokens:
        for indx, pov in enumerate(POVS):
            if tok in pov:
                counts[indx] += 1

    # assigning to the last slot is no POV pronouns were found
    if sum(counts) == 0:
        counts[-1] = 1.

    # normalizing the distribution
    norm = sum(counts)
    distr = [c / float(norm) for c in counts]

    return distr
