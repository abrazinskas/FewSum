from mltoolkit.mlmo.utils.tools import BaseConfig
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths


class BaseRunConfig(BaseConfig):
    """Contains parent parameters that are overridden in children classes."""

    def __init__(self):
        super(BaseRunConfig, self).__init__(excl_print_attrs=['exper_descr'])

        #   GENERAL  #
        self.exper_descr = 'Description of the experiment.'
        self.seed = 42
        self.cuda_device_ids = [0]
        self.training_logging_step = 100
        self.shuffler_buffer_size = 50
        self.grads_clip = 0.25
        self.strict_load = True

        #   GENERAL DATA RELATED  #
        self.dataset = 'amazon'
        self.min_rev_per_group = 9
        self.max_rev_per_group = 9

        #   DATA SOURCES  #
        self.train_early_term = None
        self.val_early_term = None

        #   DATA PATHS  #
        self.base_data_path = f"artifacts/{self.dataset}"
        self.train_fp = comb_paths(self.base_data_path, "reviews/train/")
        self.val_fp = comb_paths(self.base_data_path, 'reviews/val/')

        self.gold_train_fp = comb_paths(self.base_data_path, 'gold_summs/train.csv')
        self.gold_val_fp = comb_paths(self.base_data_path, 'gold_summs/val.csv')
        self.gold_test_fp = comb_paths(self.base_data_path, 'gold_summs/test.csv')

        self.word_vocab_fp = comb_paths(self.base_data_path,
                                        "misc/tc_word_train.txt")
        self.checkpoint_fn = 'checkpoint.tar'

        #   FREEZING AND UNFREEZING   #
        self.modules_to_unfreeze = []

        #     BPE AND TRUECASER     #
        self.subword_num = 32000
        self.bpe_fp = comb_paths(self.base_data_path,
                                 'misc/bpe_%d_train.int' % self.subword_num)
        self.bpe_vocab_fp = comb_paths(self.base_data_path,
                                       'misc/bpe_%d_train.txt' % self.subword_num)
        self.tcaser_model_path = comb_paths(self.base_data_path, 'misc', 'tcaser.model')

        #   DECODING / GENERATION  #
        self.min_seq_len = 20
        self.max_seq_len = 105
        self.seq_max_len = 105
        self.beam_size = 50
        self.block_ngram_repeat = 3
        self.ngram_mirror_window = 3
        self.mirror_conjs = ["and", "or", ",", "but"]
        self.block_consecutive = True


