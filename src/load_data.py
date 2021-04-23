
import os
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


midi_path = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_19r_step30/midi_slices'
audio_path = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_19r_step30/audio_windows'
midi_files_bin = []
audio_files_bin = []
for filename in sorted(os.listdir(midi_path)):
    midi_files_bin.append(filename)

for filename in sorted(os.listdir(audio_path)):
    audio_files_bin.append(filename)

# for midi, audio in zip(midi_files_bin, audio_files_bin):
#     print(midi[0:-14])
#     print(audio[0:-15])
#     if midi[0:-14] != audio[0:-15]:
#         print('no')

y_images = []
for filename in midi_files_bin:
    array = np.load(os.path.join(midi_path, filename))
    y_images.append(array)
y_train = np.array(y_images)
# print(y_train.shape)
# print(y_train[0])
# print(audio_files_bin)
df = pd.DataFrame()
df['filenames'] = audio_files_bin
note_labels = np.arange(21, 109)
df[note_labels] = y_train
print(df)


df_train, df_test = train_test_split(df, test_size=0.25)



gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(df_train, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                              seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,70))
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_datagen.flow_from_dataframe(df_test, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                            seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,70))
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(df_test, audio_path, x_col='filenames', batch_size=1,
                                            seed=42, shuffle=True, class_mode=None, color_mode='grayscale', target_size=(128,70))


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
# weighted_binary_crossentropy = create_weighted_binary_crossentropy(0.08, 0.92)

model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(5, 1), padding='same', input_shape=(128, 70, 1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=50, kernel_size=(3, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=50, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=25, kernel_size=(3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=25, kernel_size=(3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(352, activation='relu'))
model.add(Dense(88, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', 'binary_accuracy'])

model.fit(train_gen, steps_per_epoch=4540, epochs=10, validation_data=valid_gen,
        validation_steps=1513, verbose=1)



model.save('../models/sus_128mels_50l_19r__model')
