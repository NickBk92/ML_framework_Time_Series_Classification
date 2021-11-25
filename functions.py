import re
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random as rnd

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import np_utils

from keras import backend as K


def synthetic_data_generator(name="test_dataset", dataset_len=8000, scale=1):
    """"
    Creates a synthetic dataset with randomized parameters that will be used to train and validate the neural network architectures.\
    Outputs a pickle file that contains the dataset as a list of waveform - label pairs.

    I use autoregressive moving average process to model the noise of the accelerometer.
    I use a random sine generator to picture the head movement on 2 axis scheme.
    I use a 200 length window for our data. This is equivalent with 1 second of accelerometer data with sampling rate 200Hz
    """

    dataset_dict = dict()
    dataset_list = []
    # dataset_len = 8000
    class_size = dataset_len / 4
    signal_length = 300
    y = np.linspace(1, signal_length, signal_length)
    scale = 1
    # Event Type Generator Left, Right, NoEvent, Noise
    event_set = ("Left", "Right", "NoEvent", "Noise")
    # event_type = rnd.choice(event_set)
    event_set = ("Left", "Right", "NoEvent", "Noise")
    for i in range(0, dataset_len):
        event_type = event_set[int(i / class_size)]
        labeled_signal = np.zeros((signal_length, 1))
        if event_type == "Right":
            # Look Right Event
            # randomize the parameters
            ar0 = 4 * np.random.rand() + 6
            ar1 = 0.01 * np.random.rand()
            ar2 = 0.1 * np.random.rand()
            ar_param = np.array([ar0, ar1, ar2])
            ma_param = np.array([1, 0.1])
            # Create the base signal (ARMA process)
            AR_object1 = ArmaProcess(ar_param, ma_param)
            simulated_axis_1 = AR_object1.generate_sample(nsample=signal_length)
            # Create sine signals
            sine_length = np.random.randint(85, 125)
            possition = np.random.randint(0, signal_length - sine_length)
            sine_ampl = np.random.rand() + 1
            x = np.linspace(-np.pi, np.pi, sine_length)
            sin = sine_ampl * np.sin(x)
            simulated_axis_1[possition : possition + sine_length] += sin
            simulated_axis_1 = simulated_axis_1[50:250] * scale
        elif event_type == "Left":
            # Look Left Event
            # randomize the parameters
            ar0 = 4 * np.random.rand() + 6
            ar1 = 0.01 * np.random.rand()
            ar2 = 0.1 * np.random.rand()
            ar_param = np.array([ar0, ar1, ar2])
            ma_param = np.array([1, 0.1])
            # Create the base signal (ARMA process)
            AR_object1 = ArmaProcess(ar_param, ma_param)
            simulated_axis_1 = AR_object1.generate_sample(nsample=signal_length)
            # Create sine signals
            sine_length = np.random.randint(85, 125)
            possition = np.random.randint(0, signal_length - sine_length)
            sine_ampl = np.random.rand() + 1
            x = np.linspace(-np.pi, np.pi, sine_length)
            sin = sine_ampl * np.sin(x)
            simulated_axis_1[possition : possition + sine_length] -= sin
            simulated_axis_1 = simulated_axis_1[50:250] * scale
        elif event_type == "NoEvent":
            # No Event window
            # randomize the parameters
            ar0 = 4 * np.random.rand() + 6
            ar1 = 0.01 * np.random.rand()
            ar2 = 0.1 * np.random.rand()
            ar_param = np.array([ar0, ar1, ar2])
            ma_param = np.array([1, 0.1])
            # Create the base signal (ARMA process)
            event_type = "Noise"
            AR_object1 = ArmaProcess(ar_param, ma_param)
            simulated_axis_1 = AR_object1.generate_sample(nsample=signal_length)
            simulated_axis_1 = simulated_axis_1[50:250] * scale
        elif event_type == "Noise":
            # Noise Event

            # randomize the parameters
            ar0 = 4 * np.random.rand() + 6
            ar1 = 0.01 * np.random.rand()
            ar2 = 0.1 * np.random.rand()
            ar_param = np.array([ar0, ar1, ar2])
            ma_param = np.array([1, 0.1])
            # Create the base signal (ARMA process)
            AR_object1 = ArmaProcess(ar_param, ma_param)
            simulated_axis_1 = AR_object1.generate_sample(nsample=signal_length)
            noise_length1 = np.random.randint(14, 75)
            noise_length2 = np.random.randint(14, 75)
            possition = np.random.randint(
                0, signal_length - max(noise_length1, noise_length2)
            )

            noise_ampl = np.random.rand() + 1.5
            noise1 = noise_ampl * AR_object1.generate_sample(nsample=noise_length1)
            simulated_axis_1[possition : possition + noise_length1] -= noise1
            simulated_axis_1 = simulated_axis_1[50:250] * scale

        data = simulated_axis_1
        # dataset.update({event_type : data})
        dataset_entry = (data, event_type, labeled_signal)
        dataset_list.append(dataset_entry)
        if event_type in dataset_dict:
            dataset_dict[event_type].append(data)
        else:
            dataset_dict.update({event_type: []})
            dataset_dict[event_type].append(data)
    event_type = "Left"
    df = pd.DataFrame(dataset_list)
    df.to_pickle(name + ".pkl")


def data_loader(dataset_name="test_dataset.pkl", split_factor=0.75, batch_size=8):
    path = Path(os.getcwd())
    dataset = pd.read_pickle(path / dataset_name)
    dataset_len = len(dataset)
    checkpoint = int(split_factor * (dataset_len / 4))
    suffle_buffer_size = 50

    signal_length = len(dataset[0][0])
    num_classes = 3
    input_shape = (signal_length, 1)
    x_train = []
    y_train = []
    for i in range(0, checkpoint):
        x_train.append(dataset[0][i].T)
        y_train.append(0)
        x_train.append(dataset[0][i + int(dataset_len / 4)].T)
        y_train.append(1)
        x_train.append(dataset[0][i + 2 * int(dataset_len / 4)].T)
        y_train.append(2)
        x_train.append(dataset[0][i + 3 * int(dataset_len / 4)].T)
        y_train.append(2)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_train = x_train.reshape(-1, signal_length, 1)
    y_train = np.asarray(y_train)
    y_train = np_utils.to_categorical(y_train)

    x_test = []
    y_test = []
    for i in range(checkpoint, int(dataset_len / 4)):
        x_test.append(dataset[0][i].T)
        y_test.append(0)
        x_test.append(dataset[0][i + int(dataset_len / 4)].T)
        y_test.append(1)
        x_test.append(dataset[0][i + 2 * int(dataset_len / 4)].T)
        y_test.append(2)
        x_test.append(dataset[0][i + 3 * int(dataset_len / 4)].T)
        y_test.append(2)
    x_test = np.asarray(x_test, dtype=np.float32)
    x_test = x_test.reshape(-1, signal_length, 1)
    y_test = np.asarray(y_test)
    y_test = np_utils.to_categorical(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(suffle_buffer_size).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

class Model:

    def __init__(self,signal_length = 200,num_classes = 3):
        print('inside constr')
        self.network = keras.Sequential()
        self.network.add(layers.Conv1D(5, 20, activation='relu',input_shape=(signal_length,1)))
        self.network.add(layers.MaxPooling1D(3))
        self.network.add(layers.Conv1D(10, 10, activation='relu',input_shape=(60,5)))
        self.network.add(layers.GlobalAveragePooling1D())
        self.network.add(layers.Dense(num_classes, activation='softmax'))
        
        self.callbacks_list = [
        keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=5),
        ]

    def summary(self):
        print(self.network.summary())

    def train(self, dataset_name="test_dataset.pkl",batch_size = 100):
        EPOCHS = 100
        train_dataset, test_dataset = data_loader(dataset_name = dataset_name, batch_size = batch_size)
        self.network.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

        history = self.network.fit(train_dataset,
                      epochs=EPOCHS,
                      callbacks=self.callbacks_list,
                      verbose=0)
        return(history)
