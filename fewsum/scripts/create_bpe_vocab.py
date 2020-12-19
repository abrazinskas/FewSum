from fewsum.utils.constants import SPECIAL_TOKENS
from fewsum.utils.tools import BPE
from argparse import ArgumentParser
from functools import partial
from sacremoses import MosesTruecaser
from fewsum.data_pipelines.assemblers import assemble_vocab_pipeline
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from fewsum.utils.fields import InpDataF
from mltoolkit.mldp.utils.tools import Vocabulary


def create_bpe_vocabulary(bpe_vocab_fp, bpe_int_fp, data_path, truecaser_fp):
    """Creates vocabulary that is used to map BPEs to ids and vice-versa.

    It iterates over data, performs tokenization with the BPE tokenizer and
    true caser and creates a vocabulary of unique symbols that is saved.

    Args:
        bpe_vocab_fp: path to the new vocabulary that will be created.
        data_path: path to data with text.
        bpe_int_fp: internal file with BPEs.
        truecaser_fp: self-explanatory.
    """
    bpe = BPE(glossaries=SPECIAL_TOKENS)
    bpe.load(bpcodes_fp=bpe_int_fp, merges=-1)

    tcaser = MosesTruecaser(load_from=truecaser_fp, is_asr=True)
    tcase_func = partial(tcaser.truecase, return_str=True, use_known=True)
    unsup_tok_func = lambda x: bpe.tokenize(tcase_func(x).split())

    #   PIPELINES AND VOCAB   #

    vocab_pipeline = assemble_vocab_pipeline(text_fname=InpDataF.REV_TEXT,
                                             lowercase=False,
                                             tok_func=unsup_tok_func)
    subword_vocab = Vocabulary(vocab_pipeline, name_prefix="word",
                               special_tokens=SPECIAL_TOKENS)
    subword_vocab.create(data_source={"data_path": data_path}, max_size=None,
                         data_fnames=InpDataF.REV_TEXT)
    subword_vocab.write(bpe_vocab_fp, sep=' ')


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument("--bpe_vocab_fp", type=str)
    parser.add_argument('--bpe_int_fp', type=str)
    parser.add_argument('--data_path', type=str, nargs='+')
    parser.add_argument('--truecaser_fp', type=str)
    args = parser.parse_args()
    create_bpe_vocabulary(**vars(args))
