import os
import py_midicsv as pm
import librosa.display
import numpy as np

for filename in os.listdir('/media/zackstrater/New Volume/maestro-v3.0.0/2018'):
    if filename.endswith(".midi"):
        # # print(os.path.join(directory, filename))
        # print(filename)
        # # import midi file, result is a list of strings that contain midi actions
        # csv_string = pm.midi_to_csv(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2008', filename))
        # meta_data = csv_string[0:7]
        # track_end = csv_string[-2:]
        # header = meta_data[0].split(',')
        # tempo = meta_data[2].split(',')
        # ticks_per_quarter = int(header[-1])
        # microsec_per_quarter = int(tempo[-1])
        # microsec_per_tick = microsec_per_quarter / ticks_per_quarter
        # seconds_per_tick = microsec_per_tick / 1000000
        # song_total_ticks = int(track_end[0].split(', ')[1])
        # microsec_per_quarter = int(tempo[-1])
        # print(f'tempo: {microsec_per_quarter}')
        # print(f'PPQ: {ticks_per_quarter}')
        pass
    else:
        if filename.endswith(".wav"):
            y, sr = librosa.load(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2018', filename), sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_mel_sprectrogram = librosa.power_to_db(mel_spectrogram)
            print(np.min(log_mel_sprectrogram))
            print(np.max(log_mel_sprectrogram))

# import numpy as np
# np. set_printoptions(threshold=np. inf)
# np.set_printoptions(linewidth=100)
# array = np.arange(240).reshape(10, 24)
# print(array)
# print('\n\n')
#
#
# def windows(array, stepsize, left_buffer, right_buffer):
#     array_len = array.shape[1]
#     first_sample = left_buffer
#     last_sample = array_len - right_buffer
#     center_indexes = np.arange(first_sample, last_sample, stepsize)
#     def left_right(center, left, right):
#         return center-left, center+right + 1
#     vlr = np.vectorize(left_right)
#     output = vlr(center_indexes, left_buffer, right_buffer)
#     print(center_indexes)
#     return output
#
# def apply_window_index(array, lefts, rights):
#     window_bin = []
#     for l, r in zip(lefts, rights):
#         window_bin.append(array[:, l:r])
#     return window_bin
#
# L, R = windows(array, stepsize=3, left_buffer=5, right_buffer=2)
# print(apply_window_index(array, L, R))

import numpy as np
arr = np.arange(128)
print(arr)
arr = arr[21:109]
print(arr)
print(arr.shape)
# array = np.arange(2560).reshape(128, -1)
# print(array)
# print(array.shape)