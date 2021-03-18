import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

x, sr = librosa.load('../data/test_aduio.wav', sr=None)
# x is series with amplitude at each sample
# sr is number of samples

df = pd.DataFrame(x, columns=['amplitude'])

# index is sample number, convert to time in seconds by taking inverse
# and multiplying by the sample number
df.index = [(1/sr)*i for i in range(len(df.index))]
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'time'})
df.plot.line(x='time', y='amplitude')


# fourier transform, returns matrix with magnitude of each frequency bin (rows) for each sample
X = librosa.stft(x)

# converts amplitude to db at each time point, using np.max to place 0 db
Xdb = librosa.amplitude_to_db(abs(X))


df2 = pd.DataFrame(Xdb)


plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
A, B = np.meshgrid(df2.index, df2.columns)


# C = df2.values
# print(A.shape, B.shape, C.T.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(A, B, C.T, cmap=cm.coolwarm)
plt.show()