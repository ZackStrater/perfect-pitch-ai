import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

x, sr = librosa.load('../data/test_aduio.wav', sr=95750)
print(sr)
# x is series with amplitude at each sample
# sr is number of samples

df = pd.DataFrame(x, columns=['amplitude'])

# index is sample number, convert to time in seconds by taking inverse
# and multiplying by the sample number
# df.index = [(1/sr)*i for i in range(len(df.index))]
# df.reset_index(inplace=True)
# df = df.rename(columns = {'index':'time'})
# df.plot.line(x='time', y='amplitude')


# fourier transform, returns matrix with magnitude of each frequency bin (rows) for each sample
X = librosa.stft(x)
#  https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
# need to figure out how to convert frames to midi ticks
# can use combination of hop_length and sample rate (on librosa.load) to get right number of ticks


# converts amplitude to db at each time point, using np.max to place 0 db
Xdb = librosa.amplitude_to_db(abs(X))


df2 = pd.DataFrame(Xdb)
print(Xdb.shape)
print(Xdb)
print(df2.shape)
print(df2)

# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel' )
# plt.colorbar()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

A, B = np.meshgrid(df2.index, df2.columns)
# C = df2.values
# print(A.shape, B.shape, C.T.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(A, B, C.T, cmap=cm.coolwarm)
plt.show()