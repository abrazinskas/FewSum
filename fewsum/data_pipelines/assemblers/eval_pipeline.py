from mltoolkit.mldp.utils.constants.vocabulary import PAD, START, END, UNK
from mltoolkit.mldp import PyTorchPipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, Padder, \
    SeqWrapper
from mltoolkit.mldp.steps.transformers.field import FieldRenamer, FieldDuplicator,\
    FieldDropper
from fewsum.data_pipelines.steps import RevMapper, AmazonTransformer,\
    GoldSummRevIndxsCreator
from fewsum.utils.fields import ModelF, GoldDataF
from fewsum.data_pipelines.steps.props import SummEvalRougeKnob, SummEvalPOVProp,\
    DummyProp, SummEvalLenProp
from fewsum.data_pipelines.steps import NumpyFormatter
from csv import QUOTE_NONE

TOK_SUMM1 = 'tok_summ1'
TOK_SUMM2 = 'tok_summ2'
TOK_SUMM3 = 'tok_summ3'


def assemble_eval_pipeline(word_vocab, max_groups_per_chunk=1, tok_func=None,
                           lowercase=False):
    """Assembles a data-pipeline for eval. against gold summaries."""
    assert START in word_vocab and END in word_vocab

    reader = CsvReader(sep='\t', encoding='utf-8', engine='python',
                       chunk_size=max_groups_per_chunk, use_lists=True,
                       quating=QUOTE_NONE)

    rouge_prop = SummEvalRougeKnob(hyp_fnames=[GoldDataF.SUMM1, GoldDataF.SUMM2,
                                               GoldDataF.SUMM3],
                                   ref_fnames=GoldDataF.REVS,
                                   new_fname=ModelF.ROUGE_PROP)

    field_dupl = FieldDuplicator({GoldDataF.SUMM1: TOK_SUMM1,
                                  GoldDataF.SUMM2: TOK_SUMM2,
                                  GoldDataF.SUMM3: TOK_SUMM3})
    tokenizer = TokenProcessor(fnames=[TOK_SUMM1, TOK_SUMM2, TOK_SUMM3] + GoldDataF.REVS,
                               tok_func=tok_func, lowercase=lowercase)
    field_dropper = FieldDropper([TOK_SUMM1, TOK_SUMM2, TOK_SUMM3])

    rating_prop = DummyProp(fname=GoldDataF.PROD_ID,
                            new_fname=ModelF.RATING_PROP, fval=0.)
    len_prop = SummEvalLenProp(summ_fnames=[TOK_SUMM1, TOK_SUMM2, TOK_SUMM3],
                               rev_fnames=GoldDataF.REVS,
                               new_fname=ModelF.LEN_PROP)
    pov_prop = SummEvalPOVProp(summ_fnames=[TOK_SUMM1, TOK_SUMM2, TOK_SUMM3],
                               new_fname=ModelF.POV_PROP)

    # summaries are not converted to tokens
    vocab_mapper = VocabMapper({ModelF.REV: word_vocab})

    dataset_spec_trans = AmazonTransformer([GoldDataF.PROD_ID,
                                            GoldDataF.CAT,
                                            ModelF.ROUGE_PROP, ModelF.LEN_PROP,
                                            ModelF.RATING_PROP, ModelF.POV_PROP])

    fname_renamer = FieldRenamer({GoldDataF.PROD_ID: ModelF.GROUP_ID,
                                  GoldDataF.CAT: ModelF.CAT})

    seq_wrapper = SeqWrapper(fname=[ModelF.REV], start_el=word_vocab[START].id,
                             end_el=word_vocab[END].id)

    padder = Padder(fname=[ModelF.REV], new_mask_fname=[ModelF.REV_MASK],
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    indxs_creator = GoldSummRevIndxsCreator()

    rev_mapper = RevMapper(group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                           group_rev_mask_fname=ModelF.GROUP_REV_INDXS_MASK,
                           rev_mask_fname=ModelF.REV_MASK)

    np_formatter = NumpyFormatter([ModelF.ROUGE_PROP, ModelF.LEN_PROP,
                                   ModelF.RATING_PROP, ModelF.POV_PROP])

    pipeline = PyTorchPipeline(reader=reader, error_on_invalid_chunk=False)

    pipeline.add_step(rouge_prop)
    pipeline.add_step(rating_prop)

    # props that require tokenization
    pipeline.add_step(field_dupl)
    pipeline.add_step(tokenizer)
    pipeline.add_step(pov_prop)
    pipeline.add_step(len_prop)
    pipeline.add_step(field_dropper)

    pipeline.add_step(dataset_spec_trans)

    pipeline.add_step(vocab_mapper)

    pipeline.add_step(fname_renamer)
    pipeline.add_step(seq_wrapper)

    pipeline.add_step(padder)

    pipeline.add_step(indxs_creator)
    pipeline.add_step(rev_mapper)

    pipeline.add_step(np_formatter)

    return pipeline
