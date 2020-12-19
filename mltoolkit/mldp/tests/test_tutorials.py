from mltoolkit.mldp.pipeline import Pipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor,\
    VocabMapper, Padder
from mltoolkit.mldp.steps.transformers.field import FieldSelector
from mltoolkit.mldp.utils.helpers.nlp.token_cleaning import twitter_text_cleaner
from mltoolkit.mldp.utils.tools import Vocabulary
from mltoolkit.mldp.utils.tools.vocabulary import PAD
from mltoolkit.mldp.tutorials.steps import TwitterFilesPreprocessor,\
    FeaturesLabelsFormatter
from mltoolkit.mldp.tutorials.model import ISentiLSTM
import unittest
from nltk.tokenize import TweetTokenizer
import os


class TestTutorials(unittest.TestCase):

    def setUp(self):
        self.tutorials_path = "mltoolkit.mldp/tutorials/"

    def test_how_to_apply_run(self):

        data_path = os.path.join(self.tutorials_path,
                                 "data/tweets.csv")

        # paths where vocabs will be saved and later loaded from
        words_vocab_file_path = os.path.join(self.tutorials_path,
                                             "data/vocabs/words.txt")
        labels_vocab_file_path = os.path.join(self.tutorials_path,
                                              'data/vocabs/labels.txt')

        # creating step objects
        twitter_tokenizer = TweetTokenizer()
        preprocessor = TwitterFilesPreprocessor(input_cols_number=3,
                                                tweets_indx=2,
                                                add_header=['ids', 'labels',
                                                            'tweets'])
        csv_reader = CsvReader(sep='\t', chunk_size=30)
        fields_selector = FieldSelector(fnames=["tweets", "labels"])
        token_processor = TokenProcessor(fnames="tweets",
                                         tok_func=twitter_tokenizer.tokenize,
                                         tok_cleaning_func=twitter_text_cleaner,
                                         lowercase=True)

        # data pipeline for vocabularies creation
        vocab_data_pipeline = Pipeline(reader=csv_reader,
                                       preprocessor=preprocessor,
                                       worker_processes_num=0,
                                       name_prefix="vocabs")
        vocab_data_pipeline.add_step(fields_selector)
        vocab_data_pipeline.add_step(token_processor)

        # creating or loading vocabs
        words_vocab = Vocabulary(vocab_data_pipeline, name_prefix="words")
        words_vocab.load_or_create(words_vocab_file_path,
                                   data_source={"data_path": data_path},
                                   data_fnames="tweets")

        labels_vocab = Vocabulary(vocab_data_pipeline,
                                  name_prefix="labels")
        labels_vocab.load_or_create(labels_vocab_file_path,
                                    data_source={"data_path": data_path},
                                    data_fnames="labels")

        print(words_vocab)

        print(labels_vocab)

        print(vocab_data_pipeline)

        # extra steps for training and evaluation
        mapper = VocabMapper(field_names_to_vocabs={"tweets": words_vocab,
                                                    "labels": labels_vocab})
        padder = Padder(fname="tweets", new_mask_fname="tweets_mask",
                        pad_symbol=words_vocab[PAD].id)
        formatter = FeaturesLabelsFormatter(features_field_name="tweets",
                                            labels_field_name="labels",
                                            classes_number=len(labels_vocab))

        # building the actual pipeline
        dev_data_pipeline = Pipeline(reader=csv_reader, preprocessor=preprocessor,
                                     worker_processes_num=1, name_prefix="dev")
        dev_data_pipeline.add_step(fields_selector)
        dev_data_pipeline.add_step(token_processor)
        dev_data_pipeline.add_step(mapper)
        dev_data_pipeline.add_step(padder)
        dev_data_pipeline.add_step(formatter)

        print(dev_data_pipeline)

        epochs = 2

        i_model = ISentiLSTM(dev_data_pipeline)
        i_model.init_model(words_vocab_size=len(words_vocab), input_dim=50,
                           lstm_hidden_dim=120,
                           number_of_classes=len(labels_vocab),
                           mask_symbol=words_vocab[PAD].id)
        # print("testing before training")
        # i_model.test(data_path=data_path)
        # print("training the model")
        # for epoch in range(1, epochs + 1):
        #     print "epoch %d" % epoch
        #     i_model.train(data_path=data_path)
        #     i_model.test(data_path=data_path)


if __name__ == '__main__':
    unittest.main()
