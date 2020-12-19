from mltoolkit.mldp.utils.constants.vocabulary import PAD, START, END, UNK
from mltoolkit.mldp import PyTorchPipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, \
    SeqLenComputer, Padder, SeqWrapper
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.transformers.general import Shuffler
from mltoolkit.mldp.steps.transformers.field import FieldRenamer
from fewsum.data_pipelines.steps import RevMapper, AmazonTransformer,\
    SummMapper, GoldSummRevIndxsCreator, NumpyFormatter
from fewsum.utils.fields import ModelF, GoldDataF
from csv import QUOTE_NONE
from fewsum.data_pipelines.steps.props import SummRougeProp, DummyProp, SummLenProp,\
    POVProp


def assemble_tuning_pipeline(word_vocab, max_groups_per_batch=1, tok_func=None,
                             lowercase=False):
    """
    The pipeline yields tokenized reviews and summaries that can be used for
    training (fine-tuning of the model).
    """
    assert START in word_vocab and END in word_vocab

    reader = CsvReader(sep='\t', encoding='utf-8', engine='python',
                       chunk_size=None,
                       use_lists=True, quating=QUOTE_NONE)

    chunk_accum = ChunkAccumulator(new_size=max_groups_per_batch)

    ama_spec_trans = AmazonTransformer(fnames_to_copy=[GoldDataF.PROD_ID,
                                                       GoldDataF.CAT,
                                                       ])
    summ_mapper = SummMapper(fname=ModelF.SUMMS,
                             new_indx_fname=ModelF.SUMM_GROUP_INDX)

    token_processor = TokenProcessor(fnames=[ModelF.REV, ModelF.SUMM],
                                     tok_func=tok_func, lowercase=lowercase)

    vocab_mapper = VocabMapper({ModelF.REV: word_vocab,
                                ModelF.SUMM: word_vocab})

    fname_renamer = FieldRenamer({GoldDataF.PROD_ID: ModelF.GROUP_ID,
                                  GoldDataF.CAT: ModelF.CAT,
                                  ModelF.SUMMS: ModelF.SUMM})

    seq_wrapper = SeqWrapper(fname=[ModelF.REV, ModelF.SUMM],
                             start_el=word_vocab[START].id,
                             end_el=word_vocab[END].id)

    padder = Padder(fname=[ModelF.REV, ModelF.SUMM],
                    new_mask_fname=[ModelF.REV_MASK, ModelF.SUMM_MASK],
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    indxs_creator = GoldSummRevIndxsCreator()

    # rev_mapper = RevMapper(group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
    #                        group_rev_mask_fname=ModelF.GROUP_REV_INDXS_MASK,
    #                        rev_mask_fname=ModelF.REV_MASK)

    # props
    len_prop = SummLenProp(summ_fname=ModelF.SUMM, rev_fname=ModelF.REV,
                           group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                           summ_group_indx_fname=ModelF.SUMM_GROUP_INDX,
                           new_fname=ModelF.LEN_PROP)
    pov_prop = POVProp(text_fname=ModelF.SUMM, new_fname=ModelF.POV_PROP)
    rouge_prop = SummRougeProp(summ_fname=ModelF.SUMM, rev_fname=ModelF.REV,
                               group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                               summ_group_indx_fname=ModelF.SUMM_GROUP_INDX,
                               new_fname=ModelF.ROUGE_PROP)
    rating_prop = DummyProp(fname=ModelF.SUMM, new_fname=ModelF.RATING_PROP,
                            fval=0.)

    np_formatter = NumpyFormatter([ModelF.LEN_PROP, ModelF.RATING_PROP,
                                   ModelF.POV_PROP, ModelF.ROUGE_PROP])

    pipeline = PyTorchPipeline(reader=reader, error_on_invalid_chunk=False)

    # pipeline.add_step(shuffler)
    pipeline.add_step(chunk_accum)

    pipeline.add_step(ama_spec_trans)
    pipeline.add_step(summ_mapper)

    pipeline.add_step(fname_renamer)

    pipeline.add_step(indxs_creator)

    # props
    pipeline.add_step(rating_prop)
    pipeline.add_step(rouge_prop)

    pipeline.add_step(token_processor)

    # the props below require tokenization
    pipeline.add_step(len_prop)
    pipeline.add_step(pov_prop)

    pipeline.add_step(vocab_mapper)

    pipeline.add_step(seq_wrapper)
    pipeline.add_step(padder)

    pipeline.add_step(np_formatter)

    return pipeline
