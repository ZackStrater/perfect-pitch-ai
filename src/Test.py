


from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import tensorflow
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)

# midi_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--1.midi'
# audio_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--1.wav'
# new_song = Song(midi_filepath, audio_filepath)
# new_song.process_audio_midi_save_slices('../data', '../data', normalize_mel_spectrogram=True, apply_sus=False, apply_denoising=False, alpha=8, beta=4, n_mels=128, stepsize=20, left_buffer=100, right_buffer=27, filename='44100hz', save=False)
# for midi_win, midi, audio in zip(new_song.midi_windows, new_song.midi_slices, new_song.mel_windows):
#     new_song.compare_arrays(midi_win, np.tile(midi, (10, 1)).T, audio)


# from tensorflow import keras
# model = keras.models.load_model('/home/zackstrater/DSIclass/capstones/audio-to-midi/models/test_model')


midi_path = '/home/zackstrater/audio_midi_repository/2017_50L_10R_0,25ds_sus_step61/midi_slices'
audio_path = '/home/zackstrater/audio_midi_repository/2017_50L_10R_0,25ds_sus_step61/audio_windows'
midi_win_path = '/home/zackstrater/audio_midi_repository/2017_50L_10R_0,25ds_sus_step61/midi_windows'

midi_files_bin = []
audio_files_bin = []
midi_win_bin = []
for filename in sorted(os.listdir(midi_path)):
    midi_files_bin.append(filename)

for filename in sorted(os.listdir(audio_path)):
    audio_files_bin.append(filename)

for filename in sorted(os.listdir(midi_win_path)):
    midi_win_bin.append(filename)


# for midi, audio, midi_win in zip(midi_files_bin, audio_files_bin, midi_win_bin):
#     print(midi[0:-14])
#     print(audio[0:-15])
#     print(midi_win[0:-15])
#     if midi[0:-14] != audio[0:-15]:
#         print('no')

y_images = []
for filename in midi_files_bin[::75]:
    array = np.load(os.path.join(midi_path, filename))
    y_images.append(array)
y_train = np.array(y_images)

df = pd.DataFrame()
df['filenames'] = audio_files_bin[::75]
df['midi_winfiles'] = midi_win_bin[::75]
note_labels = np.arange(21, 109)
df[note_labels] = y_train
print(df)

for index, row in df.iterrows():
    audio_img = Image.open(f'{audio_path}/{row.iloc[0]}')
    print(row.iloc[0])
    audio_arr = asarray(audio_img)/255

    midi_img = Image.open(f'{midi_win_path}/{row.iloc[1]}')
    print(row.iloc[1])
    midi_arr = asarray(midi_img)/255

    midi_slice = row.iloc[2:].values
    midi_tile = np.tile(midi_slice, (10, 1)).T.astype('float64')

    # reshaped_audio_arr = audio_arr.reshape(1, 128, 70, 1)
    # print(reshaped_audio_arr.shape)
    # midi_pred = model.predict(reshaped_audio_arr)
    # midi_pred_tile = np.tile(midi_pred, (10, 1)).T.astype('float64')
    fig, axs = plt.subplots(1, 2, figsize=(15, 20))
    axs[0].imshow(audio_arr, aspect='auto', interpolation='nearest')
    axs[1].imshow(midi_arr, aspect='auto', interpolation='nearest')
    axs[0].set_title('Audio', size=20)
    axs[0].set_xlabel('Ticks', size=15)
    axs[0].set_ylabel('log Hz', size=15)


    axs[1].set_title('MIDI', size=20)
    axs[1].set_xlabel('Ticks', size=15)
    axs[1].set_ylabel('Note', size=15)



    # axs[2].imshow(midi_tile, aspect='auto', interpolation='nearest')
    # axs[3].imshow(midi_pred_tile, aspect='auto', interpolation='nearest')
    plt.show()

