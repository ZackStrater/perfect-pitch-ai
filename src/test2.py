import os
import py_midicsv as pm
import librosa.display


# for filename in os.listdir('/media/zackstrater/New Volume/maestro-v3.0.0/2008'):
#     if filename.endswith(".midi"):
#         # print(os.path.join(directory, filename))
#         print(filename)
#         # import midi file, result is a list of strings that contain midi actions
#         csv_string = pm.midi_to_csv(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2008', filename))
#         meta_data = csv_string[0:7]
#         track_end = csv_string[-2:]
#         header = meta_data[0].split(',')
#         tempo = meta_data[2].split(',')
#         ticks_per_quarter = int(header[-1])
#         microsec_per_quarter = int(tempo[-1])
#         microsec_per_tick = microsec_per_quarter / ticks_per_quarter
#         seconds_per_tick = microsec_per_tick / 1000000
#         song_total_ticks = int(track_end[0].split(', ')[1])
#         microsec_per_quarter = int(tempo[-1])
#         print(f'tempo: {microsec_per_quarter}')
#         print(f'PPQ: {ticks_per_quarter}')
#     else:
#         if filename.endswith(".wav"):
#             x, sr = librosa.load(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2008', filename), sr=None)
#             print(sr)

import numpy as np
np. set_printoptions(threshold=np. inf)
np.set_printoptions(linewidth=100)
array = np.arange(240).reshape(10, 24)
print(array)
print('\n\n')


def windows(array, stepsize, left_buffer, right_buffer):
    array_len = array.shape[1]
    first_sample = left_buffer
    last_Sample = array_len - right_buffer
    center_indexes = np.arange(first_sample, last_Sample, stepsize)
    def left_right(center, left, right):
        return center-left, center+right + 1
    vlr = np.vectorize(left_right)
    output = vlr(center_indexes, left_buffer, right_buffer)
    print(center_indexes)
    return output

def apply_window_index(array, lefts, rights):
    window_bin = []
    for l, r in zip(lefts, rights):
        window_bin.append(array[:, l:r])
    return window_bin

L, R = windows(array, stepsize=3, left_buffer=5, right_buffer=2)
print(apply_window_index(array, L, R))

