from mltoolkit.mldp.utils.tools import Vocabulary
from fewsum.utils.constants import SPECIAL_TOKENS
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from fewsum.data_pipelines.assemblers import assemble_vocab_pipeline
from fewsum.utils.fields import InpDataF
from sacremoses import MosesTruecaser
import argparse
from functools import partial


def create_word_vocab(vocab_fp, data_path, truecaser_fp):
    """Creates a vocabulary using a vocabulary specific pipeline."""
    tcaser = MosesTruecaser(load_from=truecaser_fp, is_asr=True)
    tcase_func = partial(tcaser.truecase, return_str=True, use_known=True)
    tok_func = lambda x: tcase_func(x).split()
    vocab_pipeline = assemble_vocab_pipeline(text_fname=InpDataF.REV_TEXT,
                                             lowercase=False, tok_func=tok_func)
    word_vocab = Vocabulary(vocab_pipeline, name_prefix="word",
                            special_tokens=SPECIAL_TOKENS)

    word_vocab.create(data_source={'data_path': data_path},
                      data_fnames=InpDataF.REV_TEXT)
    word_vocab.write(vocab_fp, sep=' ')


if __name__ == '__main__':
    logger = init_logger("")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_fp", type=str)
    parser.add_argument('--data_path', type=str, nargs='+')
    parser.add_argument('--truecaser_fp', type=str)
    args = parser.parse_args()
    create_word_vocab(**vars(args))
