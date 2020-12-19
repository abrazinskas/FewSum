from argparse import ArgumentParser
from sacremoses import MosesTruecaser
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from csv import QUOTE_NONE
import os

logger_name = os.path.basename(__file__)
logger = init_logger(logger_name)


def train_and_save_true_casing_model(input_fps, text_fname, output_fp):
    """Trains the Moses model on tokenized csv files; saves params."""
    mtr = MosesTruecaser(is_asr=True)
    reader = CsvReader(quoting=QUOTE_NONE, sep='\t', engine='python',
                       encoding='utf-8')
    texts = []
    logger.info("Loading data from: '%s'." % input_fps)
    for dc in reader.iter(data_path=input_fps):
        for du in dc.iter():
            texts.append(du[text_fname].split())
    logger.info("Loaded the data.")
    safe_mkfdir(output_fp)
    logger.info("Training the truecaser.")
    mtr.train(texts, save_to=output_fp, progress_bar=True, processes=1)
    logger.info("Done, saved the model to: '%s'." % output_fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_fps", nargs="+", help='File paths with input '
                                                       'data in the csv format.')
    parser.add_argument("--text_fname", help='The column name corresponding to '
                                             'text.')
    parser.add_argument("--output_fp", help='File path were the model should be '
                                            'saved.')
    train_and_save_true_casing_model(**vars(parser.parse_args()))
