from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten
import tensorflow
import numpy as np
import os
import matplotlib.pyplot as plt

midi_path = '/media/zackstrater/New Volume/audio_windows_midi_slices/midi_slices'
audio_path = '/media/zackstrater/New Volume/audio_windows_midi_slices/audio_windows'
midi_files_bin = []
audio_files_bin = []
midi_windows_path = '/media/zackstrater/New Volume/audio_windows_midi_slices/midi_windows'
midi_windows_bin = []

for filename in sorted(os.listdir(midi_path)):
    midi_files_bin.append(filename)

for filename in sorted(os.listdir(audio_path)):
    audio_files_bin.append(filename)

for filename in sorted(os.listdir(midi_windows_path)):
    midi_windows_bin.append(filename)


for midi, audio, midi_win in zip(midi_files_bin, audio_files_bin, midi_windows_bin):
    # print(midi[0:-9])
    # print(audio[0:-10])
    # print(midi_win[0:-9])
    if midi[0:-9] != audio[0:-10]:
        print('no')
    if midi[0:-9] != midi_win[0:-9]:
        print('no')
    if audio[0:-10] != midi_win[0:-9]:
        print('no')




X_images = []
for filename in audio_files_bin[0:100]:
    array = np.load(os.path.join(audio_path, filename))
    X_images.append(array)

X_train = np.array(X_images)#.reshape((100, 128, 128, 1))
print(X_train.shape)

y_images = []
for filename in midi_files_bin[0:100]:
    array = np.load(os.path.join(midi_path, filename))
    y_images.append(array)
y_train = np.array(y_images)
print(y_train.shape)

midiwns = []
for filename in midi_windows_bin[0:100]:
    array = np.load(os.path.join(midi_windows_path, filename))
    midiwns.append(array)
mw = np.array(midiwns)
print(mw.shape)

for audio, midi, mwin in zip(X_train, y_train, mw):
    fig, axs = plt.subplots(1, 3, figsize=(15, 20))
    axs[0].imshow(mwin)
    axs[1].imshow(np.tile(midi, (10, 1)).T)
    axs[2].imshow(audio)
    plt.show()






# gpus = tensorflow.config.experimental.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(gpus[0], True)
# img = np.zeros((128, 100))
# input_shape = (img.shape[0], img.shape[1], 1)
#
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 11), padding='same', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(filters=32, kernel_size=(3, 11), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=16, kernel_size=(3, 11), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(88, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')
#
# print(model.summary())