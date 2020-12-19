from nltk.util import ngrams as compute_ngrams
import numpy as np
from collections import OrderedDict

X_AND_X_PROP = 'seq_prop_with_x_and_x'  # repetition pattern observed in the
                                        # generated sentences
X_AND_X_COUNT = 'x_and_x_count'
X_OR_X_COUNT = 'x_or_x_count'
X_COMMA_X_COUNT = 'x_,_x_count'
UN_SENT_PROP_PROP = 'un_sent_prop'
AVG_SEQ_LEN = 'avg_seq_len'
UN_SENTS = 'un_sents'
TOTAL_SENTS = 'total_sents'
TOTAL_SEQS = 'total_seqs'
AVG_SENTS = 'avg_sents'


def ngram_seq_analysis(seqs, tokenizer, sent_splitter,
                       n_grams_to_comp=(2, 3, 4, 5)):
    """
    Performs sequence repetition analytics based on:
        1. Unique N-grams proportion
        2. Unique sentences proportion
        3. X and X pattern (e.g. good and good) - the count of detected patterns

    At the moment the analytics are mainly on the level of individual sequences.
    N-grams are computed considering sentences.

    :param seqs: list/array of sequence strings.
    :param tokenizer: function for splitting strings to list of tokens.
    :param sent_splitter: function for splitting strings to list of sentence
                          strings.
    :param n_grams_to_comp: what n-grams to consider for analysis.
    :return: dict with stats aggregated over the number of sequences.
    """
    n_gram_str_fn = lambda x: "un_%dgr_prop" % x
    seqs_sents = [sent_splitter(seq_sents_tokens) for seq_sents_tokens in seqs]
    # seqs_sents_tokens is a triple nested list
    seqs_sents_tokens = [[tokenizer(sent) for sent in sents] for sents
                         in seqs_sents]
    # for each sequence it's the number of unique n-grams / total n-grams
    stats = OrderedDict()
    for ngr in n_grams_to_comp:
        stats[n_gram_str_fn(ngr)] = []

    stats[X_AND_X_PROP] = []

    total_seq_len = 0.
    x_and_x_count = 0
    x_or_x_count = 0
    x_comma_x_count = 0

    for seq_sents_tokens in seqs_sents_tokens:

        # n-gram related statistics
        for ngr in n_grams_to_comp:
            n_grams = []
            for sent_toks in seq_sents_tokens:
                n_grams += list(compute_ngrams(sent_toks, ngr))
            avg_un_ngrams = float(len(set(n_grams))) / len(n_grams) if len(
                n_grams) > 0 else 0.
            stats[n_gram_str_fn(ngr)].append(avg_un_ngrams)

        # x and x patterns and seq lens
        tmp_x_and_x_count = 0
        for sent_toks in seq_sents_tokens:
            tmp_x_and_x_count += count_mirror_patterns(sent_toks)
            x_and_x_count += count_mirror_patterns(sent_toks, center_tok="and")
            x_or_x_count += count_mirror_patterns(sent_toks, center_tok="or")
            x_comma_x_count += count_mirror_patterns(sent_toks, center_tok=",")
            total_seq_len += len(sent_toks)

        stats[X_AND_X_PROP].append(tmp_x_and_x_count > 0)

    # computing sentence related analytics
    stats[UN_SENT_PROP_PROP] = []
    total_un_sents = 0
    total_sents = 0
    for seq_sents in seqs_sents:
        # remove the last ./!/? if it's present
        un_sents = set()
        for sent in seq_sents:
            if sent[-1] in [".", "!", "?"]:
                sent = sent[:-1]
            un_sents.add(sent)
        total_un_sents += len(un_sents)
        total_sents += len(seq_sents)
        avg_un_sents_prop = float(len(un_sents))/len(seq_sents) \
            if len(seq_sents) > 0 else 0.
        stats[UN_SENT_PROP_PROP].append(avg_un_sents_prop)

    # averaging over the number of seqs
    res = OrderedDict()
    for k, v in stats.items():
        res[k] = np.mean(v)

    # extra stats
    res[UN_SENTS] = total_un_sents
    res[TOTAL_SENTS] = total_sents
    res[AVG_SENTS] = total_sents/len(seqs)
    res[AVG_SEQ_LEN] = total_seq_len/len(seqs)
    res[X_AND_X_COUNT] = x_and_x_count
    res[X_OR_X_COUNT] = x_or_x_count
    res[X_COMMA_X_COUNT] = x_comma_x_count
    res[TOTAL_SEQS] = len(seqs)

    return res


def count_mirror_patterns(tokens, center_tok='and'):
    x_and_x_count = 0
    for i in range(len(tokens)):
        if i != 0 and i + 1 < len(tokens) and tokens[i] == center_tok and \
                tokens[i - 1] == tokens[i + 1]:
            x_and_x_count += 1
    return x_and_x_count


