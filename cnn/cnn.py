import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
import tensorflow as tf


class TB(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)


input_file = 'clean_data.csv'
file = pandas.read_csv(input_file)

x = file.Tweet
y = file.Label
seed = 334
x_train, x_validation_test, y_train, y_validation_test = train_test_split(x, y, test_size=0.02, random_state=seed)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_test, y_validation_test, test_size=0.5, random_state=seed)

model_cbow = KeyedVectors.load('./word2vec_models/model_cbow_100.word2vec')
model_sg = KeyedVectors.load('./word2vec_models/model_sg_100.word2vec')

num_words = 100000

tokenizer_100000 = Tokenizer(num_words=num_words)
tokenizer_100000.fit_on_texts(x_train)
sequences_train_100000 = tokenizer_100000.texts_to_sequences(x_train)
sequences_validation_100000 = tokenizer_100000.texts_to_sequences(x_validation)
sequences_test_100000 = tokenizer_100000.texts_to_sequences(x_test)

len_max = 0
for x in x_train:
    temp = len(x.split())
    if temp > len_max:
        len_max = temp

x_train_pad = pad_sequences(sequences_train_100000, maxlen=70, padding='post')
x_validation_pad = pad_sequences(sequences_validation_100000, maxlen=70, padding='post')
x_test_pad = pad_sequences(sequences_test_100000, maxlen=70, padding='post')

embedding_matrix_cbow = np.zeros((num_words, 100))
embedding_matrix_sg = np.zeros((num_words, 100))
embedding_matrix_cbow_sg = np.zeros((num_words, 200))
for word, rank in tokenizer_100000.word_index.items():
    if rank >= num_words:
        break
    if word in model_cbow:
        embedding_matrix_cbow[rank] = model_cbow[word]
    if word in model_sg:
        embedding_matrix_sg[rank] = model_sg[word]
    if (word in model_cbow) and (word in model_sg):
        embedding_matrix_cbow_sg[rank] = np.append(model_cbow[word], model_sg[word])

model_cnn_trainable_true = Sequential()
model_cnn_trainable_true.add(Embedding(100000, 200, weights=[embedding_matrix_cbow_sg], input_length=70, trainable=True))
model_cnn_trainable_true.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_trainable_true.add(GlobalMaxPooling1D())
model_cnn_trainable_true.add(Dense(256, activation='relu'))
model_cnn_trainable_true.add(Dense(128, activation='relu'))
model_cnn_trainable_true.add(Dense(1, activation='sigmoid'))
model_cnn_trainable_true.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model_cnn_trainable_true.fit(x_train_pad, y_train, validation_data=(x_validation_pad, y_validation), epochs=5, batch_size=32, verbose=1, callbacks=[TB(log_dir='./tmp/log/model_cnn_trainable_true/')])
model_cnn_trainable_true.save('./models/model_cnn_trainable_true.h5')

model_cnn_conv2d = Sequential()
model_cnn_conv2d.add(Embedding(100000, 200, weights=[embedding_matrix_cbow_sg], input_length=70, trainable=False))
model_cnn_conv2d.add(Reshape((-1, 200, 1)))
model_cnn_conv2d.add(Conv2D(filters=70, kernel_size=(2, 2), padding='valid', activation='relu', strides=(1, 1)))
model_cnn_conv2d.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model_cnn_conv2d.add(Conv2D(filters=50, kernel_size=(2, 2), padding='valid', activation='relu', strides=(1, 1)))
model_cnn_conv2d.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model_cnn_conv2d.add(Conv2D(filters=30, kernel_size=(2, 2), padding='valid', activation='relu', strides=(1, 1)))
model_cnn_conv2d.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model_cnn_conv2d.add(Flatten())
model_cnn_conv2d.add(Dense(256, activation='relu'))
model_cnn_conv2d.add(Dense(128, activation='relu'))
model_cnn_conv2d.add(Dense(1, activation='sigmoid'))
model_cnn_conv2d.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model_cnn_conv2d.fit(x_train_pad, y_train, validation_data=(x_validation_pad, y_validation), epochs=5, batch_size=32, verbose=1, callbacks=[TB(log_dir='./tmp/log/model_cnn_conv2d/')])
model_cnn_conv2d.save('./models/model_cnn_conv2d.h5')

input_type = Input(shape=(70,), dtype='float64')
tweet_encoder = Embedding(100000, 200, weights=[embedding_matrix_cbow_sg], input_length=70, trainable=False)(input_type)
model_2 = Conv1D(filters=50, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
model_2 = MaxPooling1D(pool_size=18)(model_2)
model_2 = Flatten()(model_2)
model_3 = Conv1D(filters=50, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
model_3 = MaxPooling1D(pool_size=18)(model_3)
model_3 = Flatten()(model_3)
model_5 = Conv1D(filters=50, kernel_size=5, padding='valid', activation='relu', strides=1)(tweet_encoder)
model_5 = MaxPooling1D(pool_size=18)(model_5)
model_5 = Flatten()(model_5)
merged = concatenate([model_2, model_3, model_5], axis=1)
merged = Dense(256, activation='relu')(merged)
merged = Dense(128, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(merged)
model_cnn_mixed = Model(inputs=[input_type], outputs=[output])
model_cnn_mixed.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_mixed.fit(x_train_pad, y_train, validation_data=(x_validation_pad, y_validation), epochs=10, batch_size=32, verbose=1, callbacks=[TB(log_dir='./tmp/log/model_cnn_mixed/')])
model_cnn_mixed.save('./models/model_cnn_mixed.h5')
