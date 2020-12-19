import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, Input, Masking, Lambda, LSTM
np.random.seed(41)


class SentiLSTM(object):
    """Keras implementation of an LSTM twitter sentiment classifier."""

    def __init__(self, words_vocab_size, input_dim, lstm_hidden_dim,
                 number_of_classes, mask_symbol=None):
        self.words_vocab_size = words_vocab_size
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.number_of_classes = number_of_classes
        self.mask_symbol = mask_symbol
        self._model = self.__build()

    def __build(self):
        """Builds and returns the actual Keras model."""
        model = Sequential()
        model.add(Embedding(self.words_vocab_size, self.input_dim))
        if self.mask_symbol:
            model.add(Masking(mask_value=self.mask_symbol))
        model.add(LSTM(self.lstm_hidden_dim))
        model.add(Dense(self.number_of_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam")
        return model

    def train(self, tweets, labels):
        return self._model.train_on_batch(tweets, labels)

    def predict(self, tweets):
        return np.argmax(self._model.predict(tweets), axis=1)
