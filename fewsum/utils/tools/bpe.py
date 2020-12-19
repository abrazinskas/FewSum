from __future__ import unicode_literals, division
from subword_nmt.apply_bpe import encode, read_vocabulary, isolate_glossary
from subword_nmt.learn_bpe import learn_bpe
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir

import argparse
import os
import sys
import re

from logging import getLogger

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class BPE(object):
    """Byte pair encoding wrapper."""

    def __init__(self, separator='@@', glossaries=None):
        self.vocab = None  # need to perform a better investigation to understand if it's really useful
        self.bpe_codes = None
        self.bpe_codes_reverse = None
        self.separator = separator
        self.version = None
        self.glossaries = glossaries if glossaries else []
        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) \
            if glossaries else None
        self.cache = {}

    def load(self, bpcodes_fp, merges=-1):
        bpcodes = open(bpcodes_fp, encoding='utf-8')
        bpcodes.seek(0)
        offset = 1
        # check version information
        firstline = bpcodes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$', '',
                                                         firstline.split()[
                                                             -1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            bpcodes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item)
                          in enumerate(bpcodes) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: '
                                 '{1}\n'.format(i + offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword'
                                 ' units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in
                               reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair)
                                       for pair, i in self.bpe_codes.items()])
        logger.info("Loaded BPE from: '%s'." % bpcodes_fp)

    def create(self, vocab_fp, symbol_num, output_fp, min_vocab_freq=1):
        logger.info("Creating BPE with `symbol_num`=%d, `min_vocab_freq`=%d"
                    " from: '%s'." % (symbol_num, min_vocab_freq, vocab_fp))
        safe_mkfdir(output_fp)
        learn_bpe(infile=open(vocab_fp, mode='r', encoding='utf-8'),
                  num_symbols=symbol_num, is_dict=True,
                  min_frequency=min_vocab_freq,
                  outfile=open(output_fp, mode='w', encoding='utf-8'))
        logger.info("Saved BPE to: '%s'." % output_fp)

    def load_or_create(self, bpcodes_fp, symbol_num, vocab_fp=None,
                       min_vocab_freq=1):
        """
        Args:
            bpcodes_fp: file path with BPEs.
            symbol_num: number of merges to create or load.
            vocab_fp: vocabulary of words from which create subwords.
            min_vocab_freq: minimum frequency of words to consider during BPE
                creation.
        """
        if not os.path.isfile(bpcodes_fp):
            assert symbol_num is not None
            self.create(vocab_fp=vocab_fp, symbol_num=symbol_num,
                        output_fp=bpcodes_fp, min_vocab_freq=min_vocab_freq)
        self.load(bpcodes_fp, merges=symbol_num)

    def tokenize(self, tokens, dropout=0):
        """Segments a sequence of tokens with BPE encoding."""
        # because I'm not working in NMT, I don't use a shared vocabulary
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def detokenize(self, subwords):
        """DeSegments a sequence of subwords into tokens."""
        tokens = []
        curr_token = ""
        for subword in subwords:
            if subword[len(subword) - 2:] == self.separator:
                curr_token += subword[:-2]
            else:
                curr_token += subword
                tokens.append(curr_token)
                curr_token = ""
        return tokens

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment,
                                                                      gloss)]
        return word_segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--num_symbols", type=int, default=32000)
    parser.add_argument("--min_vocab_freq", type=int, default=2)
    parser.add_argument("--separator", type=str, default="@@")
    parser.add_argument("--glossaries", type=str, nargs="*")
    args = parser.parse_args()

    logger = init_logger("bpe")
    bpe = BPE(glossaries=args.glossaries, separator=args.separator)
    bpe.create(vocab_fp=args.vocab_fp, symbol_num=args.num_symbols,
               output_fp=args.output_fp, min_vocab_freq=args.min_vocab_freq)
