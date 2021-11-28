import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random as rnd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy import signal
from os.path import exists

import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
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

def decode(input_array):
    input_len = len(input_array)

    encoding = dict()
    decoding = dict()
    # One Hot encoding
    # Left = 1000
    key = np.array2string(np.array([1, 0, 0]))
    encoding.update({key: "Left"})
    # Right = 0100
    key = np.array2string(np.array([0, 1, 0]))
    encoding.update({key: "Right"})
    # NoEvent = 0010
    key = np.array2string(np.array([0, 0, 1]))
    encoding.update({key: "NoEvent"})


    one_hot_results = np.zeros((input_len, 3)).astype(int)
    y_results = []
    y_labels = []
    for i in range(0, input_len):
        one_hot_results[i, np.argmax(input_array[i])] = 1
        y_results.append(encoding[np.array2string(one_hot_results[i, :])])

    return y_results

def plot_metrics(result, target,title):

    y_result = decode(result)
    if type(target[0][0][0]) == np.float32:
        y_target = decode(target)
    else:
        y_target = target
    plt.figure(title)
    # classif_report = (sk.metrics.classification_report(y_target, y_result, labels=["Left", "Right", "NoEvent"], output_dict=True))
    conf_matr = (confusion_matrix(y_result, y_target, labels=["Left", "Right", "NoEvent"]))
    labels = ["Left", "Right", "NoEvent"]
    title = title
    cm_ax = sns.heatmap(conf_matr, annot=True,annot_kws={'size':16}, cmap='Blues', fmt="",square = True, xticklabels=labels, yticklabels=labels, cbar=False)
    cm_ax.xaxis.set_ticks_position('top')
    cm_ax.set_xticklabels(cm_ax.get_xmajorticklabels(), fontsize = 14)
    cm_ax.set_yticklabels(cm_ax.get_ymajorticklabels(), fontsize = 14)
    plt.title(title,fontsize=20)
    plt.ylabel('Ground Truth',fontsize=16)
    plt.xlabel('Prediction',fontsize=16)
    plt.tight_layout()
    # plt.show()

    # plt.figure()
    # title = "Classification Report"
    # metrics = ["precision", "recall", "f1-score"]
    # classif_metrics = np.zeros((3, 3))
    # for i, label in enumerate(labels):
    #     for j, metric in enumerate(metrics):
    #         classif_metrics[i, j] = classif_report[label][metric]
    #
    # cr_ax = sns.heatmap(classif_metrics, annot=True, cmap='Blues', fmt="", yticklabels=labels, xticklabels=metrics,
    #                     cbar=False)
    # cr_ax.xaxis.set_ticks_position('top')
    # plt.title(title)
    # plt.show()


class Model:
    '''
    This is the basic model class. Contains the network initialization, training, and validation.
    '''
    def __init__(self,signal_length = 200,num_classes = 3):
        self.existing_model = False
        self.path = Path(os.getcwd())
        print('Check if a model (model.h5) already exists in models folder.\n\tYES: Load the existing model \n\tNO: Create and train a new one')
        if exists(self.path/'models/model.h5'):
            print('YES: model exists\n')
            self.existing_model = True
            self.network = tf.keras.models.load_model(self.path/'models/model.h5')
        else:
            print('NO: model does not exist\n')
            self.network = keras.Sequential()
            self.network.add(layers.Conv1D(5, 20, activation='relu',input_shape=(signal_length,1)))
            self.network.add(layers.MaxPooling1D(3))
            self.network.add(layers.Conv1D(10, 10, activation='relu',input_shape=(60,5)))
            self.network.add(layers.GlobalAveragePooling1D())
            self.network.add(layers.Dense(num_classes, activation='softmax'))
        


    def summary(self):
        '''
        Prints network's details.
        '''
        print(self.network.summary())

    def train(self, dataset_name="test_dataset.pkl",batch_size = 100):
        '''
        Given a compatible dataset, trains and saves the best model.
        '''

        if self.existing_model:
            print('An trained model already exists.\nSkipping Training')
            return 0

        model_path =(self.path/"models/model.h5")

        self.callbacks_list = [
                        keras.callbacks.ModelCheckpoint(
                        filepath=model_path,
                        monitor='val_loss',
                        save_best_only=False),
                        keras.callbacks.EarlyStopping(monitor='accuracy', patience=2),
                        ]
        EPOCHS = 100
        train_dataset,_ = data_loader(dataset_name = dataset_name, batch_size = batch_size)
        self.network.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

        history = self.network.fit(train_dataset,
                      epochs=EPOCHS,
                      callbacks=self.callbacks_list,
                      verbose=2)
        return(history)
        
    def validate(self,dataset_name="test_dataset.pkl"):
        '''
        Prints Confusion Matrix of model's performance given a compatible dataset.
        '''
        _,test_dataset = data_loader(dataset_name = dataset_name, batch_size = 1)
        target = []; [target.append(i[1].numpy()) for i in test_dataset]
        self.results = self.network.predict(test_dataset)
        plot_metrics(self.results,target,'Model Validation Results')

    def insights(self):
        dataset = pd.read_pickle(self.path/"test_dataset.pkl")
        sample_input_left = dataset[0][8].reshape(1,-1,1)
        sample_input_right = dataset[0][2004].reshape(1,-1,1)
        sample_input_noevent = dataset[0][7004].reshape(1,-1,1)

        ### PLOT First Layer Results Per Kernel Per Class  ####
        layers = []
        for l in self.network.layers:
            layers.append(l)

        first_layer = layers[0]
        out_left = first_layer(sample_input_left)
        out_right = first_layer(sample_input_right)
        out_noevent = first_layer(sample_input_noevent)

        plt.figure('First Layer Analysis')
        plt.rc('ytick',labelsize=10)
        plt.rc('xtick',labelsize=10)
        plt.subplot(6,3,1)
        plt.title('Left',fontsize = 14)
        plt.plot(sample_input_left[0,:,0])
        plt.ylim(-1.8,1.8)
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
        plt.subplot(6,3,2)
        plt.title('Right',fontsize = 14)
        plt.plot(sample_input_right[0,:,0])
        plt.ylim(-1.8,1.8)
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
        plt.subplot(6,3,3)
        plt.title('Noise',fontsize = 14)
        plt.plot(sample_input_noevent[0,:,0])
        plt.ylim(-1.8,1.8)
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
        for i in range(5):
            plt.subplot(6,3,3*(i+1)+1)
            plt.plot(out_left[0,:,i])
            plt.xticks([])
            plt.subplot(6,3,3*(i+1)+2)
            plt.plot(out_right[0,:,i])
            plt.xticks([])
            plt.subplot(6,3,3*(i+1)+3)
            plt.plot(out_noevent[0,:,i])
            plt.xticks([])
        # plt.tight_layout()




        layers = []
        for l in self.network.layers:
            layers.append(l)
        first_layer = layers[0]

        #### Plot Kernel Frequency Repsonse #####
        sr = 200
        plt.figure('Kernel Frequency Responce')
        plt.rc('ytick',labelsize=10)
        plt.title('Frequency response',verticalalignment = 'center_baseline')
        for i in range(5):
            weights =np.flip(first_layer.weights[0][:,0,i].numpy().reshape(20))
            w, h = signal.freqz(b=weights.T, a=1)
            x = w * sr * 1.0 / (2 * np.pi)
            y = 20 * np.log10(abs(h))
            # plt.figure(figsize=(10,5))
            plt.subplot(5,1,i+1)
            plt.semilogx(x, y)
            if i == 4:
                plt.xticks([0.5,1,5, 10, 20, 50, 100], ["0,5","1","5", "10", "20", "50", "100"])
            else:
                plt.xticks([5, 10, 20, 50, 100, 200],['','','','','',''])
            plt.yticks([-20,-10,0,10,20],['-20','-10','0','10','20'])
            plt.grid(which='both', linestyle='-', color='grey')
            plt.xlim(0,180)
            plt.ylim(-25,25)
            plt.ylabel('Amplitude [dB]',fontsize=16) if i==2 else 0
        plt.xlabel('Frequency [Hz]',fontsize=16)
        # plt.show()