from mltoolkit.mldp import PyTorchPipeline
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.collectors import UnitSampler, ChunkCollector, ChunkShuffler
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.general import Postfixer, ChunkSorter
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, \
    SeqLenComputer, SeqWrapper, Padder
from mltoolkit.mldp.steps.transformers.field import FieldRenamer
from mltoolkit.mldp.utils.constants.vocabulary import START, END, PAD
from fewsum.data_pipelines.steps import GroupRevIndxsCreator, RevMapper, FieldMerger, \
    UniqueWordCalc,  NumpyFormatter, TextLenFilter
from fewsum.data_pipelines.steps.props import LenProp, POVProp
from mltoolkit.mldp.steps.preprocessors import FileShuffler
from fewsum.utils.fields import InpDataF, ModelF
from csv import QUOTE_NONE


def assemble_unsup_pipeline(word_vocab, max_groups_per_batch=1, reader_threads=5,
                            min_revs_per_group=None, max_revs_per_group=10,
                            worker_num=1, seed=None, tok_func=None,
                            lowercase=True, max_len=None,
                            shuffler_buffer_size=250):
    """Creates a data-pipeline that yields batches for to train the unsup. model.

    Creates a flow of data transformation steps that modify the data until the
    final form is reached in terms of PyTorch tensors.

    Args:
        word_vocab: vocabulary object with words/tokens.
        max_groups_per_batch: number of groups each batch should have.
        min_revs_per_group: number of reviews a group should have in order not
            to be discarded.
        max_revs_per_group: self-explanatory.
        seed: used to use the same data subsamples/shuffles every epoch.
        max_len: if passed will filter out all reviews that a longer than the
            threshold.
    Returns:
        DataPipeline object that allows iteration over batches/chunks.
    """
    assert START in word_vocab and END in word_vocab

    file_shuffler = FileShuffler()

    # TODO: explain how grouping works here - each file has reviews of a group

    reader = CsvReader(sep='\t', engine='c', chunk_size=None,
                       encoding='utf-8', quoting=QUOTE_NONE, buffer_size=200,
                       timeout=None, worker_threads_num=reader_threads,
                       use_lists=True)

    fname_renamer = FieldRenamer({InpDataF.REV_TEXT: ModelF.REV,
                                  InpDataF.GROUP_ID: ModelF.REV_GROUP_ID,
                                  InpDataF.RATING: ModelF.REV_RATING,
                                  InpDataF.RATING_DEV: ModelF.RATING_PROP,
                                  InpDataF.CAT: ModelF.REV_CAT})

    unit_sampler = UnitSampler(id_fname=ModelF.REV_GROUP_ID, sample_all=True,
                               min_units=min_revs_per_group,
                               max_units=max_revs_per_group)

    unit_sampler_accum = ChunkAccumulator(unit_sampler)

    # since we're splitting one group into multiple chunks, it's convenient
    # to postfix each group_id name, such that it would be possible to
    # associate summaries with different subsets of reviews
    postfixer = Postfixer(id_fname=ModelF.REV_GROUP_ID)

    # property and related steps
    len_prop = LenProp(len_fname=ModelF.REV_LEN, new_fname=ModelF.LEN_PROP)
    pov_prop = POVProp(text_fname=ModelF.REV, new_fname=ModelF.POV_PROP)

    rouge_field_merger = FieldMerger(merge_fnames=[InpDataF.ROUGE1,
                                                   InpDataF.ROUGE2,
                                                   InpDataF.ROUGEL],
                                     new_fname=ModelF.ROUGE_PROP)

    # to avoid having same product/business appearing in the same merged
    # data-chunk, buffer a small number of them, shuffle, and release
    chunk_shuffler = ChunkAccumulator(ChunkShuffler(buffer_size=shuffler_buffer_size))

    # accumulates a fixed number of group chunks, merges them
    # together, and passes along the pipeline
    chunk_coll = ChunkCollector(buffer_size=max_groups_per_batch, strict=True)
    chunk_accum = ChunkAccumulator(chunk_coll)

    # alternation of data entries
    tokenizer = TokenProcessor(fnames=ModelF.REV, tok_func=tok_func,
                               lowercase=lowercase)
    vocab_mapper = VocabMapper({ModelF.REV: word_vocab})

    seq_wrapper = SeqWrapper(fname=ModelF.REV,
                             start_el=word_vocab[START].token,
                             end_el=word_vocab[END].token)

    seq_len_computer = SeqLenComputer(ModelF.REV, ModelF.REV_LEN)

    padder = Padder(fname=ModelF.REV,
                    new_mask_fname=ModelF.REV_MASK,
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    summ_rev_indxs_creator = GroupRevIndxsCreator(
        rev_group_id_fname=ModelF.REV_GROUP_ID,
        rev_cat_fname=ModelF.REV_CAT)

    rev_mapper = RevMapper(group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                           group_rev_mask_fname=ModelF.GROUP_REV_INDXS_MASK,
                           rev_mask_fname=ModelF.REV_MASK)

    # extra steps for the loss associated with probability mass
    un_word_cal = UniqueWordCalc(new_fname=ModelF.OTHER_REV_UWORDS,
                                 rev_fname=ModelF.REV,
                                 other_rev_indxs_fname=ModelF.OTHER_REV_INDXS,
                                 other_rev_indxs_mask_fname=ModelF.OTHER_REV_INDXS_MASK)
    un_word_padder = Padder(fname=ModelF.OTHER_REV_UWORDS,
                            new_mask_fname=ModelF.OTHER_REV_UWORDS_MASK,
                            pad_symbol=word_vocab[PAD].id, padding_mode='right')

    numpy_formatter = NumpyFormatter(fnames=[ModelF.ROUGE_PROP, ModelF.RATING_PROP,
                                             ModelF.LEN_PROP, ModelF.POV_PROP])

    pipeline = PyTorchPipeline(reader=reader, preprocessor=file_shuffler,
                               worker_processes_num=worker_num,
                               seed=seed, output_buffer_size=50,
                               error_on_invalid_chunk=False, timeout=None)

    pipeline.add_step(fname_renamer)
    pipeline.add_step(rouge_field_merger)
    pipeline.add_step(tokenizer)

    if max_len:
        pipeline.add_step(TextLenFilter(fname=ModelF.REV, max_len=max_len))

    pipeline.add_step(unit_sampler_accum)
    pipeline.add_step(postfixer)

    pipeline.add_step(chunk_shuffler)

    pipeline.add_step(seq_wrapper)
    pipeline.add_step(seq_len_computer)

    # properties
    pipeline.add_step(len_prop)
    pipeline.add_step(pov_prop)

    pipeline.add_step(chunk_accum)
    pipeline.add_step(vocab_mapper)
    pipeline.add_step(padder)

    # adding additional fields for attention and summarization
    pipeline.add_step(summ_rev_indxs_creator)
    pipeline.add_step(rev_mapper)

    # adding steps for word count computation
    pipeline.add_step(un_word_cal)
    pipeline.add_step(un_word_padder)

    pipeline.add_step(numpy_formatter)

    return pipeline
