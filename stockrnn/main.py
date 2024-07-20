import tensorflow as tf
import numpy as np
import csv
import keras as keras

class RNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.rnn = keras.layers.LSTM(8, activation='tanh', recurrent_activation='relu')
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(4, activation='sigmoid')
    def call(self, input):
        x = self.rnn(input)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Agent:
    def __init__(self, lr=0.003, beta_1=0.9, beta_2=0.999, timesteps=10, batch_size=4, epochs=50):
        self.model = RNNModel()
        self.batch_size = batch_size
        self.epochs = epochs
        self.timesteps = timesteps
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2), loss=keras.losses.MSE, metrics=[keras.metrics.MeanSquaredError()])
        self.data, self.labels = self.generate_data(steps=timesteps)
    def generate_data(self, steps=10):
        data = []
        with open('appledata.csv', mode = 'r') as file:
            file_data = csv.reader(file)
            for line in file_data:
                data.append(line)
        data.pop(0)
        network_data = np.flip(np.array(data, dtype=np.float32), axis=0)
        network_data = network_data/np.max(network_data, axis=0, keepdims=True)
        pre_batched = []
        pre_labels = []
        for i in range(0, np.size(network_data, axis=0)-steps):
            pre_batched.append(network_data[i:i+steps])
            pre_labels.append(network_data[i+steps-1])
        return np.array(pre_batched, dtype=np.float32), np.array(pre_labels, dtype=np.float32)

agent = Agent()
agent.model.fit(agent.data, agent.labels, batch_size=agent.batch_size, epochs=agent.epochs)