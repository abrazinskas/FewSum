from mltoolkit.mldp import Pipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor
from csv import QUOTE_NONE


def assemble_vocab_pipeline(text_fname, sep='\t', encoding='utf-8',
                            tok_func=None, lowercase=True):
    """Assembler for the vocabulary pipeline based on a CSV reader."""
    reader = CsvReader(sep=sep, encoding=encoding, quoting=QUOTE_NONE)
    token_processor = TokenProcessor(fnames=text_fname, lowercase=lowercase,
                                     tok_func=tok_func)
    # creating vocabulary pipeline
    vocab_pipeline = Pipeline(reader=reader)
    vocab_pipeline.add_step(token_processor)
    return vocab_pipeline
