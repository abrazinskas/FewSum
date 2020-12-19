from .senti_lstm import SentiLSTM
import numpy as np


class ISentiLSTM(object):
    """Interface to access the Keras LSTM for sentence sentiment analysis."""

    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline
        self.model = None  # will be initialized in init_model

    def init_model(self, **kwargs):
        """Initializes the actual Keras model."""
        self.model = SentiLSTM(**kwargs)

    def train(self, **data_source_kwargs):
        """Trains the model for a single epoch."""
        itr = self.data_pipeline.iter(**data_source_kwargs)
        for counter, (tweets_batch, labels_batch) in enumerate(itr, 1):
            loss = self.model.train(tweets_batch, labels_batch)
            if counter % 100 == 0:
                print ("chunk's # %d loss: %f" % (counter, loss))

    def test(self, **data_source_kwargs):
        """Iterates over data batches, computes and prints accuracy [0, 1]."""
        correct = 0
        total = 0
        itr = self.data_pipeline.iter(**data_source_kwargs)
        for tweets_batch, labels_batch in itr:
            predictions = self.model.predict(tweets_batch)
            correct += np.sum(predictions == np.argmax(labels_batch, axis=1))
            total += len(tweets_batch)
        print ("accuracy: %f" % (float(correct)/total))
