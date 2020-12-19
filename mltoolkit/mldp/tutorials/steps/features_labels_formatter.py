from mltoolkit.mldp.steps.formatters.base_formatter import BaseFormatter
import numpy as np


class FeaturesLabelsFormatter(BaseFormatter):
    """Formats batches into features and one-hot encoded labels tuple."""

    def __init__(self, features_field_name, labels_field_name, classes_number):
        super(FeaturesLabelsFormatter, self).__init__()
        self.feature_field_name = features_field_name
        self.labels_field_name = labels_field_name
        self.classes_number = classes_number

    def _format(self, data_chunk):
        features = data_chunk[self.feature_field_name]
        lbls = data_chunk[self.labels_field_name]
        labels = np.eye(self.classes_number, dtype="float32")[lbls]
        return features, labels
