
import os
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D, Activation, Flatten, Dropout, Input, Bidirectional, LSTM, Reshape
from keras.models import load_model, Model
from keras                 import backend as K

from sklearn.model_selection import train_test_split
from PIL import Image

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
melsVal = np.zeros((1, 20, 200))

cptDir = 'Magenta chk'
lstmWidth = 256
inputs = Input(shape=(melsVal.shape[1], melsVal.shape[2]))

ConvBnRelu = lambda n: lambda x: Activation('relu')(BatchNormalization(scale=False)(Conv2D(n, 3, padding='same', use_bias=False)(x)))
outputs = MaxPool2D((1, 2))(ConvBnRelu(96)(MaxPool2D((1, 2))(ConvBnRelu(48)(ConvBnRelu(48)(Reshape((melsVal.shape[1], melsVal.shape[2], 1))(inputs))))))

model = Model(inputs, Dense(88, activation='sigmoid')(Bidirectional(LSTM(lstmWidth,
    recurrent_activation='sigmoid', implementation=2, return_sequences=True, unroll=True))(Dense(768, activation='relu')(
    Reshape((K.int_shape(outputs)[1], K.int_shape(outputs)[2] * K.int_shape(outputs)[3]))(outputs)))))

print(model.summary())