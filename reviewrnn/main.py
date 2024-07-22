import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import keras
import string
from keras import ops
import os


os.environ['KERAS_BACKEND'] = 'tensorflow'
keras.utils.set_random_seed(692024)

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0, from_logits=False, axis=-1):
    if from_logits: 
        y_pred = ops.softmax(y_pred, axis)
    epsilon = keras.backend.epsilon()
    y_pred = ops.clip(y_pred, x_min=epsilon, x_max=1.0-epsilon)
    p_t = y_true*y_pred+(1.0-y_true)*(1.0-y_pred)
    factor = (1.0 - p_t) ** gamma
    bce = -(y_true*ops.log(y_pred)+(1.0-y_true)*ops.log(1.0-y_pred))
    return ops.sum(factor*bce, axis=axis, keepdims=False)

def gcu(x):
    return x * ops.cos(x)

class GLU(keras.layers.Layer):
    def __init__(self, units=32, activation=keras.activations.sigmoid, **kwargs):
        super().__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units*2), # we're going to split the doubled layer in half to retain dims
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units*2,), initializer=keras.initializers.Zeros(), trainable=True
        )
    def call(self, input):
        a1 = ops.matmul(input, self.w) + self.b
        # the * sign for hadamard product when combining the two halves
        return a1[:, :self.units] * self.activation(a1[:, self.units:])


def extract_text_and_label(data):
    text = data['text']
    label = data['label']
    return (text, label)

class RNNModel(keras.Model):
    def __init__(self, vocab_size, max_len):
        super().__init__()
        '''
        GRU stuff
        U_z, U_r, U, W_z, W_r, W, are all trainable variables
        update = sigmoid(U_z*x_t + W_z*h_(t-1))
        reset (aka r_t) = sigmoid(U_r*x_t + W_r*h_(t-1)) --> how much to forget
        mem = tanh(U*x_t + r_t * h_(t-1)) --> this is the new memory of the cell, but r_t product is element wise
        h_t (aka hidden state) = update * h_(t-1) + (1 - update) * mem --> all products are element wise here
        '''
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = keras.layers.Embedding(vocab_size, 50)
        self.rnn = keras.layers.GRU(32, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences=True) # need a 3d input
        self.rnn2 = keras.layers.GRU(64, activation='tanh', recurrent_activation='sigmoid')
        self.dense1 = GLU(128)
        self.dense2 = keras.layers.Dense(1, activation='sigmoid')
    def call(self, input):
        x = self.embedding(input)
        x = self.rnn(x)
        x = self.rnn2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Agent:
    def __init__(self, lr=0.003, beta_1=0.9, beta_2=0.999, batch_size=4, epochs=50, max_len=100, max_vocab_size=8000, **kwargs):
        self.model = RNNModel(max_vocab_size, max_len)
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size
        self.datasets = tfds.load('imdb_reviews')
        self.train_dataset = tf.data.Dataset.map(self.datasets['train'], extract_text_and_label)
        self.test_dataset = tf.data.Dataset.map(self.datasets['test'], extract_text_and_label)
        self.vocab, self.inv_vocab, self.x_train, self.y_train, self.x_test, self.y_test = self.build_data()
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2), loss=focal_loss, metrics=['accuracy'])
    def index(self, vocab, para):
        text = list(map(lambda word : vocab[word] if word in vocab else vocab['UNKNOWN_WORD'], para))
        if len(text) < self.max_len:
            return text + [0] * (self.max_len-len(text))
        elif len(text) > self.max_len:
            return text[:self.max_len]
        return text
    def build_data(self):
        vocab = {'<pad>' : 0, 'UNKNOWN_WORD' : 1}
        freq = {'<pad>' : 1000000, 'UNKNOWN_WORD' : 1000000}
        x_train, x_test = [], []
        y_train, y_test = np.zeros(shape=(len(self.train_dataset),), dtype=np.float32), np.zeros(shape=(len(self.test_dataset),), dtype=np.float32)
        # training dataset data
        for x in self.train_dataset:
            review = str(x[0].numpy())[2:-1].lower()
            tokenized = review.translate(str.maketrans('', '', string.punctuation)).split()
            for token in tokenized:
                if token not in freq:
                    freq[token] = 1
                else:
                    freq[token] += 1
            x_train.append(tokenized)
            y_train[len(x_train)-1] = float(x[1])
        # do the same for testing data
        for x in self.test_dataset:
            review = str(x[0].numpy())[2:-1].lower()
            tokenized = review.translate(str.maketrans('', '', string.punctuation)).split()
            for token in tokenized:
                if token not in freq:
                    freq[token] = 1
                else:
                    freq[token] += 1
            x_test.append(tokenized)
            y_test[len(x_test)-1] = float(x[1])
        # clean up the vocabulary
        freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
        inv_vocab = {}
        count = 0
        for word in freq:
            if(count < self.max_vocab_size):
                vocab[word] = count
                inv_vocab[count] = word
                count += 1
            else:
                break
        # turn the text into indices
        x_train = [self.index(vocab, para) for para in x_train]
        x_test = [self.index(vocab, para) for para in x_test]
        return vocab, inv_vocab, np.array(x_train, dtype=np.float32), y_train, np.array(x_test, dtype=np.float32), y_test
            
# overfitting may be a problem maybe bc i'm stacking these GRUs idk, but maybe that's fixable with diff design
agent = Agent(lr=0.001, beta_1=0.9, beta_2=0.999, batch_size=32, epochs=10, max_len=100, max_vocab_size=8000)
print(f"\033[93mmemory usage of training data: {(agent.x_train.nbytes + agent.y_train.nbytes) / 1024 ** 2} megabytes\033[00m")
agent.model.fit(agent.x_train, agent.y_train, batch_size=agent.batch_size, epochs=agent.epochs, validation_data=(agent.x_test, agent.y_test))