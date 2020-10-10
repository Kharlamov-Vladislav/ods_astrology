import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('../datasets nature/nature_computer_science.csv')
corpus = df['title']

tokenizer = Tokenizer()


def get_sequence_of_tokens(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(corpus)


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'), dtype='int16')

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words, dtype='int16')
    return predictors, label, max_sequence_len


predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    model.add(Embedding(total_words, 10, input_length=input_len))

    model.add(LSTM(100))
    model.add(Dropout(0.1))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


lstm_model = create_model(max_sequence_len, total_words)
lstm_model.summary()
with tf.device('/gpu:0'):
    lstm_model.fit(predictors, label, epochs=250, verbose=5)




