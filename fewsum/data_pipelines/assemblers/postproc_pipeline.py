from mltoolkit.mldp import Pipeline
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.collectors import UnitSampler
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.general import Postfixer, FunctionApplier
from mltoolkit.mldp.steps.transformers.field import FieldDuplicator, FieldDropper
from fewsum.data_pipelines.steps.props import RougeProp, RatingProp
from fewsum.data_pipelines.steps import TypeConverter
from fewsum.utils.fields import InpDataF
from csv import QUOTE_NONE


def assemble_postproc_pipeline(text_prep_func, min_revs_per_group=None, seed=None,
                               max_revs_per_group=10, rouge_kwargs=None,
                               workers=1):
    """Creates a data-pipeline that yields batches with computed ROUGE score.

    Args:
        min_revs_per_group: number of reviews a group should have in order not
            to be discarded.
        max_revs_per_group: self-explanatory.
        seed: used to use the same data subsamples/shuffles every epoch.
    Returns:
        DataPipeline object that allows iteration over batches/chunks.
    """

    reader = CsvReader(sep='\t', engine='c', chunk_size=None,
                       encoding='utf-8', quoting=QUOTE_NONE, use_lists=True,
                       timeout=None, worker_threads_num=1)

    converter = TypeConverter(fname=InpDataF.RATING, dtype_func=float,
                              remove_invalid=True)

    unit_sampler = UnitSampler(id_fname=InpDataF.GROUP_ID, sample_all=True,
                               min_units=min_revs_per_group,
                               max_units=max_revs_per_group)

    unit_sampler_accum = ChunkAccumulator(unit_sampler)

    # since we're splitting one group into multiple chunks, it's convenient
    # to postfix each group_id name, such that it would be possible to
    # associate summaries with different subsets of reviews
    postfixer = Postfixer(id_fname=InpDataF.GROUP_ID)

    field_dupl = FieldDuplicator({InpDataF.REV_TEXT: "dummy"})

    # the field below is needed as I'm detokenizing (with text_prep_func)
    # before computing ROUGE.
    func_appl = FunctionApplier({'dummy': text_prep_func})

    # props
    rouge_prop = RougeProp(rev_fname='dummy', rouge_kwargs=rouge_kwargs)
    rating_prop = RatingProp(rating_fname=InpDataF.RATING,
                             new_fname=InpDataF.RATING_DEV)

    field_dropper = FieldDropper('dummy')

    pipeline = Pipeline(reader=reader, worker_processes_num=workers,
                        seed=seed, output_buffer_size=40,
                        error_on_invalid_chunk=True, timeout=None)
    pipeline.add_step(converter)
    pipeline.add_step(unit_sampler_accum)
    pipeline.add_step(postfixer)
    pipeline.add_step(field_dupl)
    pipeline.add_step(func_appl)
    # props
    pipeline.add_step(rouge_prop)
    pipeline.add_step(field_dropper)
    pipeline.add_step(rating_prop)

    return pipeline
