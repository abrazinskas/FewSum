from argparse import ArgumentParser
from fewsum.data_pipelines.assemblers import assemble_postproc_pipeline
import numpy as np
import os
from fewsum.utils.fields import InpDataF
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkdir, comb_paths
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from time import time
from sacremoses import MosesDetokenizer

seed = 42


def postprocess_data(data_path, out_dir_path, min_revs_per_file=None, workers=1,
                     max_revs_per_file=9, early_term=None, logging_period=1000):
    """
    Creates `K` reviews per group files, computes ROUGE 1 vs rest. In this case,
    avoids an expensive online computation of ROUGE.
    """
    logger = init_logger("", output_path=os.path.dirname(out_dir_path))
    dt = MosesDetokenizer()
    detok_func = lambda x: [dt.detokenize(_x.split(" "), unescape=False)
                            for _x in x]
    data_pipeline = assemble_postproc_pipeline(text_prep_func=detok_func, seed=seed,
                                               min_revs_per_group=min_revs_per_file,
                                               max_revs_per_group=max_revs_per_file,
                                               workers=workers)
    logger.info("Writing chunks to: '%s'." % out_dir_path)
    safe_mkdir(out_dir_path)
    chunks_count = 0
    start = time()
    unique_groups = set()
    review_count = 0
    min_rev_per_chunk = float('inf')
    max_rev_per_chunk = float('-inf')
    for dc in data_pipeline.iter(data_path=data_path, early_term=early_term):
        assert len(np.unique(dc[InpDataF.GROUP_ID])) == 1
        group_id = dc[0, InpDataF.GROUP_ID].split("_")[0]
        unique_groups.add(group_id)
        review_count += len(dc)
        min_rev_per_chunk = min(min_rev_per_chunk, len(dc))
        max_rev_per_chunk = max(max_rev_per_chunk, len(dc))
        fp = comb_paths(out_dir_path, "%s.csv" % dc[0][InpDataF.GROUP_ID])
        dc.to_csv(open(fp, encoding='utf-8', mode='w'))
        chunks_count += 1
        if chunks_count % logging_period == 0:
            logger.info("Wrote %d chunks." % chunks_count)
    logger.info("Totally wrote %d chunks." % chunks_count)
    logger.info("Total time elapsed: %f." % (time() - start))
    logger.info("Unique groups: %d." % len(unique_groups))
    logger.info("Total reviews: %d." % review_count)
    logger.info("Min reviews per chunk: %d." % min_rev_per_chunk)
    logger.info("Max reviews per chunk: %d." % max_rev_per_chunk)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--out_dir_path')
    parser.add_argument('--min_revs_per_file', type=int, default=None)
    parser.add_argument('--max_revs_per_file', type=int, default=9)
    parser.add_argument('--early_term', type=int, default=None)
    parser.add_argument('--logging_period', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=1)
    postprocess_data(**vars(parser.parse_args()))
