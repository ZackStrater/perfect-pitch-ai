
import os
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


midi_path = '/media/zackstrater/New Volume/audio_windows_midi_slices/ms'
audio_path = '/media/zackstrater/New Volume/audio_windows_midi_slices/aw'
midi_files_bin = []
audio_files_bin = []
for filename in sorted(os.listdir(midi_path)):
    midi_files_bin.append(filename)

for filename in sorted(os.listdir(audio_path)):
    audio_files_bin.append(filename)

for midi, audio in zip(midi_files_bin, audio_files_bin):
    # print(midi[0:-9])
    # print(audio[0:-10])
    # print(midi_win[0:-9])
    if midi[0:-13] != audio[0:-14]:
        print('no')

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
# print(df)


train_datagen = ImageDataGenerator()
train_gen = train_datagen.flow_from_dataframe(df, audio_path, x_col='filenames', y_col=note_labels, batch_size=32, seed=42, shuffle=True, class_mode='raw')



