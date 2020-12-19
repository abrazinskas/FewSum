from mltoolkit.mldp.steps.preprocessors import BasePreProcessor
from mltoolkit.mlutils.helpers.general import listify
from mltoolkit.mlutils.helpers.paths_and_files import get_file_paths, safe_mkdir
from mltoolkit.mldp.utils.helpers.validation import validate_data_paths
import os
import codecs
import re


class TwitterFilesPreprocessor(BasePreProcessor):
    """
    Performs a static cleaning of tweets that should be performed on new
    new datasets that have the same traits.

    E.g. it eliminates input file lines that do not have the correct number of
    components. It can happen when erroneous double \n is present.

    See _clean_and_save_file for more details on procedure specifics.
    """

    def __init__(self, input_cols_number, output_folder='data/clean_tweets/',
                 input_sep='\t', output_sep='\t', add_header=None,
                 tweets_indx=None, encoding='utf-8', forced_rerun=False):
        """
        :param input_cols_number: number of components csv input files have.
        :param input_sep: self-explanatory.
        :param output_sep: self-explanatory.
        :param add_header: list of str or None.
        :param tweets_indx: index where the tweets reside. Basic cleaning
                            of strings will be applied.
        :param encoding: self-explanatory.
        :param forced_rerun: if set to True cleaning will be re-executed
                            regardless that output files are found in the output
                            folder from a previous run.
        """
        super(TwitterFilesPreprocessor, self).__init__()
        self.input_cols_number = input_cols_number
        self.output_folder = output_folder
        self.input_sep = input_sep
        self.output_sep = output_sep
        self.add_header = add_header
        self.tweets_indx = tweets_indx
        self.encoding = encoding

    def __call__(self, data_path):
        try:
            validate_data_paths(data_path)
        except Exception as e:
            raise e

        safe_mkdir(self.output_folder)
        for dp in listify(data_path):
            for file_path in get_file_paths(dp):
                file_name = os.path.basename(file_path)
                output_file_path = os.path.join(self.output_folder, file_name)
                if not os.path.exists(output_file_path):
                    self._clean_and_save_file(file_path, output_file_path)
        return {"data_path": self.output_folder}

    def _clean_and_save_file(self, input_file_path, output_file_path):
        """Cleans a column-based file(csv), and saved the output separately."""
        input_file = codecs.open(input_file_path, encoding=self.encoding)
        output_file = codecs.open(output_file_path, 'w', encoding=self.encoding)

        # 1. write header to the output file if needed
        if self.add_header:
            output_file.write(self.output_sep.join(self.add_header) + "\n")

        # 2. strip lines and clean line parts if needed
        for i, line in enumerate(input_file):
            line = line.strip()
            if line == "":
                continue
            parts = re.split(self.input_sep, line)

            # 3. throw away invalid lines
            if len(parts) != self.input_cols_number:
                continue

            if self.tweets_indx:
                # 4. some tweets have double qoutes around, which is not
                # necessary
                tweet = parts[self.tweets_indx]
                if len(tweet) > 1 and tweet[0] == '\"' and tweet[-1]:
                    parts[self.tweets_indx] = tweet[1:-1]
            output_file.write(self.output_sep.join(parts) + "\n")

        input_file.close()
        output_file.close()
