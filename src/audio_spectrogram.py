import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

y, sr = librosa.load('/media/zackstrater/New Volume/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.wav', sr=None)
print(sr)
# x is series with amplitude at each sample
# sr is number of samples


mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=200)
log_mel_sprectrogram = librosa.power_to_db(mel_spectrogram)

X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))

plt.grid(b=None)
plt.imshow(np.flipud(Xdb), aspect='auto', interpolation='nearest')
plt.show()


plt.grid(b=None)
plt.imshow(np.flipud(log_mel_sprectrogram), aspect='auto', interpolation='nearest')
plt.show()

#
# # index is sample number, convert to time in seconds by taking inverse
# # and multiplying by the sample number
# # df.index = [(1/sr)*i for i in range(len(df.index))]
# # df.reset_index(inplace=True)
# # df = df.rename(columns = {'index':'time'})
# # df.plot.line(x='time', y='amplitude')
#

#
#
# # fourier transform, returns matrix with magnitude of each frequency bin (rows) for each sample
# X = librosa.stft(x)
# #  https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
# # need to figure out how to convert frames to midi ticks
# # can use combination of hop_length and sample rate (on librosa.load) to get right number of ticks
#
#
# # converts amplitude to db at each time point, using np.max to place 0 db
# Xdb = librosa.amplitude_to_db(abs(X))
#
#
# df2 = pd.DataFrame(Xdb)
# print(Xdb.shape)
# print(Xdb)
# from scipy.ndimage import zoom
# flipped_Xdb = np.flipud(Xdb)
# cut_Xdb = flipped_Xdb[700:, :]
# formatted_Xdb = zoom(flipped_Xdb, (1, (1/8.177)), order=0)
# print(formatted_Xdb.shape)
# plt.grid(b=None)
# plt.imshow(cut_Xdb, aspect='auto')
# plt.show()
#
# def split_into_intervals(array, time_interval, sample_rate, hop_length=512):
#     # get sample ticks per time interval
#     ticks_per_interval = time_interval * sample_rate / hop_length
#     subarray_length = ticks_per_interval
#     print(f'subarray length {subarray_length}')
#
#     array_length = array.shape[1]
#     print(f'array length {array_length}')
#     # number of columns left over
#     last_interval_length = array_length % subarray_length
#     print(f'mod {last_interval_length}')
#     # how many columns to add to make another full interval
#     padding = subarray_length - last_interval_length
#     print(f'padding {padding}')
#     padded_array = np.pad(array, ((0, 0), (0, int(padding))))
#     padded_array_length = padded_array.shape[1]
#
#     num_intervals = padded_array_length / subarray_length
#     print(f'num intervals {num_intervals}')
#     # split array into subsections ultimately based on time_interval and sample_rate
#     split_arrays = np.array_split(padded_array, num_intervals, axis=1)
#     print(f'split array shape = {split_arrays[0].shape}')
#     print(f'split array shape = {split_arrays[-1].shape}')
#     print(len(split_arrays))
#     return split_arrays
#
# split_arrays = split_into_intervals(Xdb, 8, 48000)
#
#
# print(df2.shape)
# print(df2)
#
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()
#
# # plt.figure(figsize=(14, 5))
# # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel' )
# # plt.colorbar()
# # plt.show()
# #
# #
# # plt.figure(figsize=(14, 5))
# # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log' )
# # plt.colorbar()
# # plt.show()
# #
# #
# # plt.figure(figsize=(14, 5))
# # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='cqt_hz' )
# # plt.colorbar()
# # plt.show()
#
#
#
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
#
# A, B = np.meshgrid(df2.index, df2.columns)
# # C = df2.values
# # print(A.shape, B.shape, C.T.shape)
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(A, B, C.T, cmap=cm.coolwarm)
