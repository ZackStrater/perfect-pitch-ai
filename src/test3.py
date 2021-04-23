
# import numpy as np
# arr = np.array(
# [[0.33007777, 0.3134178, 0.30268776, 0.3618889,  0.34385043, 0.36136377],
# [0.3804329,  0.3911069, 0.37348753, 0.43376762, 0.44231623, 0.408857],
# [0.32265222, 0.37470865, 0.3798349, 0.39321297, 0.4560908, 0.43245733]])
#
# print(arr)
#
#
# def db_sigmoid(x):
#     return 1 / (1 + np.exp(-alpha * x - beta))
#
#
# vectorized_db_sigmoid = np.vectorize(db_sigmoid)
# vectorized_db_sigmoid(self.mel_spectrogram)

import pandas as pd
import numpy as np
df = pd.read_csv('../data/sus_128mels_100l_27r_step20.csv')
df = df.iloc[:, 1:]
print(df.shape)
fractions = df.sum()/263167
print(fractions)
print(np.average(fractions))
print(np.max(fractions))

