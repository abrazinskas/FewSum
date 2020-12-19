from fewsum.config.run import *
from fewsum.config.model_hp import *
from fewsum.modelling.models import *
from fewsum.data_pipelines.assemblers import assemble_unsup_pipeline, \
    assemble_vocab_pipeline, assemble_eval_pipeline, assemble_tuning_pipeline
from mltoolkit.mldp.utils.tools import Vocabulary
from mltoolkit.mldp.utils.constants.vocabulary import START, END, PAD, UNK
from mltoolkit.mlutils.helpers.logging_funcs import init_logger, DEBUG, INFO
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths
from fewsum.utils.fields import InpDataF, ModelF
from torch.nn.init import xavier_uniform_, normal_
from mltoolkit.mlutils.helpers.formatting.general import format_big_box
from fewsum.utils.tools.seq_post_processor import SeqPostProcessor
from time import time
from torch import manual_seed
import numpy as np
import os
from fewsum.utils.constants import SPECIAL_TOKENS
import nltk
from functools import partial
from fewsum.utils.tools import BPE
from sacremoses import MosesTruecaser
from mltoolkit.mlmo.utils.helpers.pytorch.general import count_trainable_parameters, \
    count_parameters, freeze_parameters
from sacremoses import MosesTokenizer, MosesDetokenizer
from mltoolkit.mlmo.utils.tools import ModuleParallel
from fewsum.modelling.generators import Beamer
from fewsum.modelling.interfaces import IDevSumm as IDev
from fewsum.modelling.interfaces import ISumm as IModel
from fewsum.utils.constants import UNSUP, NOV_RED, PLUGIN_INIT, PLUGIN_TUNING, JOINT_TUNING
from fewsum.utils.helpers.registries import MODEL_REGISTRY, RUN_CONFIG_REGISTRY, \
    HP_CONFIG_REGISTRY
from argparse import ArgumentParser

#   PARSER   #

parser = ArgumentParser()
parser.add_argument('--regime', type=str, help='Sets the regime of training/inference.', required=True)
parser.add_argument('--inference', action='store_true',
                    help='If set, will perform inference/summary generation otherwise training.')

regime = parser.parse_args().regime
inference = parser.parse_args().inference

run_conf = RUN_CONFIG_REGISTRY[regime]()

logger = init_logger(logger_name="", level=INFO,
                     output_path=comb_paths(run_conf.output_path, "log.txt"))

#   ENV and hyper-params handling  #
manual_seed(run_conf.seed)
np.random.seed(run_conf.seed)
cuda_visible_devices = str(run_conf.cuda_device_ids) \
    if isinstance(run_conf.cuda_device_ids, int) else \
    ",".join([str(dev_id) for dev_id in run_conf.cuda_device_ids])
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
device_count = 1 if not isinstance(run_conf.cuda_device_ids, list) \
    else max(1, len(run_conf.cuda_device_ids))
device = 'cuda' if len(run_conf.cuda_device_ids) > 0 else 'cpu'
logger.info('CUDA_VISIBLE_DEVICES=%s' % cuda_visible_devices)

#   DATA SOURCES   #

vocab_data_source = {"data_path": run_conf.train_fp}
uns_train_data_source = {"data_path": run_conf.train_fp,
                         "early_term": run_conf.train_early_term}
uns_val_data_source = {"data_path": run_conf.val_fp,
                       'early_term': run_conf.val_early_term}

# PLEASE NOTE: summary training/eval does not work for multi-gpu setup at the moment
gold_train_data_source = {'data_path': run_conf.gold_train_fp} \
    if device_count == 1 else None
gold_val_data_source = {"data_path": run_conf.gold_val_fp} \
    if device_count == 1 else None
gold_test_data_source = {"data_path": run_conf.gold_test_fp} \
    if device_count == 1 else None

# registering them
DATA_SOURCE_REGISTRY = {UNSUP: {'train': uns_train_data_source,
                                'val': uns_val_data_source},
                        NOV_RED: {'train': uns_train_data_source,
                                  'val': uns_val_data_source},
                        PLUGIN_INIT: {'train': uns_train_data_source,
                                      'val': uns_val_data_source},
                        PLUGIN_TUNING: {'train': gold_train_data_source,
                                        'val': gold_val_data_source},
                        JOINT_TUNING: {'train': gold_train_data_source,
                                       'val': gold_val_data_source}
                        }

#   TRUECASER   #

tcaser = MosesTruecaser(load_from=run_conf.tcaser_model_path, is_asr=True)
tcase_func = partial(tcaser.truecase, return_str=True, use_known=True)


#   WORD TOKENIZERS / DE-TOKENIZERS   #

mt = MosesTokenizer()
dt = MosesDetokenizer()

#   SUB-WORD TOKENIZER   #

bpe = BPE(glossaries=SPECIAL_TOKENS)
bpe.load(bpcodes_fp=run_conf.bpe_fp)

unsup_tok_func = lambda x: bpe.tokenize(tcase_func(x).split())
gold_tok_func = lambda x: bpe.tokenize(mt.tokenize(tcase_func(x), escape=False))
detok_func = lambda x: dt.detokenize(bpe.detokenize(x), unescape=False)

#   DATA PIPELINES AND VOCAB   #

vocab_pipeline = assemble_vocab_pipeline(text_fname=InpDataF.REV_TEXT,
                                         lowercase=False, tok_func=unsup_tok_func)
subword_vocab = Vocabulary(vocab_pipeline, name_prefix="word",
                           special_tokens=SPECIAL_TOKENS)
subword_vocab.load(run_conf.bpe_vocab_fp, max_size=None, sep=' ')
subword_vocab.write(comb_paths(run_conf.output_path,
                               "bpe_%d_vocab.txt" % run_conf.subword_num), sep=' ')
worker_num = 3 * device_count
reader_threads = 3 * device_count
uns_train_pipeline = assemble_unsup_pipeline(word_vocab=subword_vocab,
                                             reader_threads=reader_threads,
                                             worker_num=worker_num,
                                             max_len=run_conf.max_seq_len,
                                             max_groups_per_batch=run_conf.train_groups_per_batch,
                                             min_revs_per_group=run_conf.min_rev_per_group,
                                             max_revs_per_group=run_conf.max_rev_per_group,
                                             lowercase=False, tok_func=unsup_tok_func,
                                             seed=None,
                                             shuffler_buffer_size=run_conf.shuffler_buffer_size)

uns_val_pipeline = assemble_unsup_pipeline(word_vocab=subword_vocab,
                                           worker_num=worker_num,
                                           reader_threads=reader_threads,
                                           max_len=run_conf.max_seq_len,
                                           max_groups_per_batch=run_conf.val_groups_per_batch,
                                           min_revs_per_group=run_conf.min_rev_per_group,
                                           max_revs_per_group=run_conf.max_rev_per_group,
                                           lowercase=False, tok_func=unsup_tok_func,
                                           seed=run_conf.seed,
                                           shuffler_buffer_size=run_conf.shuffler_buffer_size)

gold_train_pipeline = assemble_tuning_pipeline(subword_vocab, tok_func=gold_tok_func,
                                               lowercase=False,
                                               max_groups_per_batch=run_conf.train_groups_per_batch)
# the high-level difference between two is that the latter is used for ROUGE
# computation while the former for internal stats, such as log-likelihood.
gold_val_pipeline = assemble_tuning_pipeline(subword_vocab, tok_func=gold_tok_func,
                                             lowercase=False,
                                             max_groups_per_batch=run_conf.val_groups_per_batch)
gold_eval_pipeline = assemble_eval_pipeline(subword_vocab, tok_func=gold_tok_func,
                                            lowercase=False,
                                            max_groups_per_chunk=run_conf.eval_groups_per_batch)
# registering data pipelines
PIPELINE_REGISTRY = {
    UNSUP: {'train': uns_train_pipeline, 'val': uns_val_pipeline},
    NOV_RED: {'train': uns_train_pipeline, 'val': uns_val_pipeline},
    PLUGIN_INIT: {'train': uns_train_pipeline, 'val': uns_val_pipeline},
    PLUGIN_TUNING: {'train': gold_train_pipeline, 'val': gold_val_pipeline},
    JOINT_TUNING: {'train': gold_train_pipeline, 'val': gold_val_pipeline}
}

train_pipeline = PIPELINE_REGISTRY[regime]['train']
val_pipeline = PIPELINE_REGISTRY[regime]['val']


#   MODEL AND INTERFACES INITIALIZATION   #

model_hp = HP_CONFIG_REGISTRY[regime]()
model_class = MODEL_REGISTRY[regime]

#   - because BPEs are used, the vocabulary size is set dynamically -   #
setattr(model_hp, 'word_emb_num', len(subword_vocab))

start_id = subword_vocab[START].id
end_id = subword_vocab[END].id
pad_id = subword_vocab[PAD].id

model = model_class(**model_hp.to_dict())

#   PARAMETERS FREEZING AND UNFREEZING   #

if run_conf.modules_to_unfreeze:
    # freeze_parameters(model)
    logger.info(f"Frozen all parameters except: "
                f"{','.join(run_conf.modules_to_unfreeze)}.")
    freeze_parameters(model, blacklist=run_conf.modules_to_unfreeze)

#   PARALLELIZATION OVER MULTIPLE GPUs   #
if regime in [PLUGIN_TUNING, JOINT_TUNING] and device_count > 1:
    raise ValueError("At the moment, tuning on summaries works only on a single GPU.")

if device_count > 1 and device == 'cuda':
    model = ModuleParallel(model)

gen_func = Beamer(decoding_func=model.decode,
                  start_id=start_id, end_id=end_id,
                  device=device, len_norm=False, beam_size=run_conf.beam_size,
                  block_ngram_repeat=run_conf.block_ngram_repeat,
                  ngram_mirror_window=run_conf.ngram_mirror_window,
                  mirror_conj_ids=[subword_vocab[conj].id for conj in
                                   run_conf.mirror_conjs],
                  block_consecutive=run_conf.block_consecutive)

imodel = IModel(model=model, gen_func=gen_func, grads_clip=run_conf.grads_clip,
                learning_rate=run_conf.learning_rate, device=device)

#   - post-processors for sequence formatting -   #
gen_seq_postproc = SeqPostProcessor(word_vocab=subword_vocab,
                                    detokenizer=detok_func,
                                    sent_splitter=nltk.sent_tokenize,
                                    retain_end_token=False)
eval_postproc = SeqPostProcessor(sent_splitter=nltk.sent_tokenize)

idev = IDev(imodel=imodel, train_data_pipeline=train_pipeline,
            val_data_pipeline=val_pipeline,
            eval_data_pipeline=gold_eval_pipeline,
            word_vocab=subword_vocab, seq_postproc=gen_seq_postproc,
            summ_postproc=eval_postproc)

#   PARAMETERS LOADING OR INITIALIZATION   #

if run_conf.checkpoint_path:
    imodel.load_state(run_conf.checkpoint_path, strict=run_conf.strict_load)
else:
    imodel.init_weights(multi_dim_init_func=xavier_uniform_,
                        single_dim_init_func=lambda x: normal_(x, std=0.1))

idev.save_setup_str(run_conf.output_path, run_conf.exper_descr)

# logging and saving hyper-params
logger.info(format_big_box(str(run_conf)))
logger.info(format_big_box(str(model_hp)))
tr_par, tot_par = count_trainable_parameters(model), count_parameters(model)
logger.info(f"Trainable parameters: {tr_par}/{tot_par} ({100 *tr_par/tot_par:.2f} %).")
run_conf.save(comb_paths(run_conf.output_path, 'run_conf.json'))
model_hp.save(comb_paths(run_conf.output_path, 'model_hp.json'))

# the function executed after each epoch
after_ep_func = idev.after_ep_wrapper(run_conf.output_path,
                                      summ_eval_data_source=gold_val_data_source,
                                      checkpoint_fn=run_conf.checkpoint_fn,
                                      summ_eval_kwargs={
                                          'min_seq_len': run_conf.min_seq_len,
                                          'max_seq_len': run_conf.max_seq_len,
                                          'use_true_props': regime in [UNSUP, NOV_RED]})

if run_conf.epochs > 0 and not inference:
    #   TRAINING PROCEDURE  #
    start = time()
    try:
        idev.standard_workflow(train_data_source=DATA_SOURCE_REGISTRY[regime]['train'],
                               val_data_source=DATA_SOURCE_REGISTRY[regime]['val'],
                               logging_period=run_conf.training_logging_step,
                               epochs=run_conf.epochs,
                               after_epoch_func=after_ep_func)
    except Exception as e:
        logger.error(e)
        raise e
    logger.info("Total time elapsed: %f (s) " % (time() - start))
else:
    #   INFERENCE PROCEDURE   #
    idev.after_ep_wrapper(run_conf.output_path,
                          summ_eval_data_source=gold_test_data_source,
                          summ_eval_kwargs={
                              'min_seq_len': run_conf.min_seq_len,
                              'max_seq_len': run_conf.max_seq_len,
                              'use_true_props': regime in [UNSUP, NOV_RED]}
                          )()
