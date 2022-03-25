import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam


def get_rnn_model():
    model = Sequential()
    model.add(Embedding(187, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    opt = Adam(0.001)

    model.compile(loss=sparse_categorical_crossentropy, optimizer=opt, metrics=['acc'])
    model.summary()
    return model

def main():
    df_train = pd.read_csv("../data/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("../data/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    model = get_rnn_model()
    model.fit(X, Y, epochs=100, verbose=2, validation_split=0.1)

if __name__ == '__main__':
    main()
