
import os
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def train_model(audio_path, midi_path, epochs=20, batch_size=96, filters1=48, filters2=96, save=False, model_name=''):

    midi_files_bin = []
    audio_files_bin = []
    for filename in sorted(os.listdir(midi_path)):
        midi_files_bin.append(filename)

    for filename in sorted(os.listdir(audio_path)):
        audio_files_bin.append(filename)


    y_images = []
    for filename in midi_files_bin:
        array = np.load(os.path.join(midi_path, filename))
        y_images.append(array)
    y_train = np.array(y_images)


    df = pd.DataFrame()
    df['filenames'] = audio_files_bin
    note_labels = np.arange(21, 109)
    df[note_labels] = y_train



    img = Image.open(f'{audio_path}/{df.iloc[0, 0]}')
    img_arr = np.asarray(img)
    input_rows = img_arr.shape[0]
    input_columns = img_arr.shape[1]
    print(input_rows, input_columns)


    df_train, df_test = train_test_split(df, test_size=0.20)

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_dataframe(df_train, audio_path, x_col='filenames', y_col=note_labels, batch_size=batch_size,
                                                  seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(input_rows,input_columns))
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_gen = valid_datagen.flow_from_dataframe(df_test, audio_path, x_col='filenames', y_col=note_labels, batch_size=batch_size,
                                                seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(input_rows,input_columns))







    model = Sequential()
    model.add(Conv2D(input_shape=(input_rows, input_columns, 1), filters=filters1, kernel_size=(3, 3), padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=filters1, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=filters2, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=filters2, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(352, activation='relu'))
    model.add(Dense(352, activation='relu'))
    model.add(Dense(88, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam',
                  metrics=['accuracy', 'binary_accuracy',
                           tensorflow.keras.metrics.Precision(),
                           tensorflow.keras.metrics.Recall(),
                           ])

    print(model.summary())
    model.fit(train_gen, steps_per_epoch=df_train.shape[0] / batch_size, epochs=epochs, validation_data=valid_gen,
              validation_steps=df_test.shape[0] / batch_size, verbose=1)

    if save:
        model.save(f'../models/{model_name}')




midi_path = '/home/zackstrater/audio_midi_repository/200mel_10L_9R_0,5ds_NOsus_step20/midi_slices'
audio_path = '/home/zackstrater/audio_midi_repository/200mel_10L_9R_0,5ds_NOsus_step20/audio_windows'
train_model(audio_path, midi_path, epochs=6, save=True, model_name='200mel_10L_9R_0,5ds_NOsus_step20_epoch6')



# import keras.backend as K
#
# def create_weighted_binary_crossentropy(zero_weight, one_weight):
#
#     def weighted_binary_crossentropy(y_true, y_pred):
#
#         # Original binary crossentropy (see losses.py):
#         # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
#
#         # Calculate the binary crossentropy
#         b_ce = K.binary_crossentropy(y_true, y_pred)
#
#         # Apply the weights
#         weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
#         weighted_b_ce = weight_vector * b_ce
#
#         # Return the mean error
#         return K.mean(weighted_b_ce)
#
#     return weighted_binary_crossentropy
#
# weighted_binary_crossentropy = create_weighted_binary_crossentropy(1, 10)

