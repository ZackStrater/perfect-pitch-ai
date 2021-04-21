from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten
import tensorflow
import numpy as np
import os


# midi_path = '/media/zackstrater/New Volume/split_audio_midi_files/midi_files'
# audio_path = '/media/zackstrater/New Volume/split_audio_midi_files/audio_files'
# midi_files_bin = []
# audio_files_bin = []
#
# for filename in sorted(os.listdir(midi_path)):
#     midi_files_bin.append(filename)
#
# for filename in sorted(os.listdir(audio_path)):
#     audio_files_bin.append(filename)
#
#
# for midi, audio in zip(midi_files_bin, audio_files_bin):
#     if midi[0:-12] != audio[0:-13]:
#         print(midi[0:-12])
#         print(audio[0:-13])
#
# X_images = []
# for filename in audio_files_bin[0:100]:
#     array = np.load(os.path.join(audio_path, filename))
#     X_images.append(array)
#
# X_train = np.array(X_images).reshape((100, 200, 750, 1))
# print(X_train.shape)
#
# y_images = []
# for filename in midi_files_bin[0:100]:
#     array = np.load(os.path.join(midi_path, filename))
#     y_images.append(array)
# y_train = np.array(y_images).reshape((100, 100, 750, 1))
# print(y_train.shape)
#
#
#
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
img = np.zeros((128, 100))
input_shape = (img.shape[0], img.shape[1], 1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 11), padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 11), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 11), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(88, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

print(model.summary())