


import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)

pic = Image.open("/media/zackstrater/New Volume/testing_png/audio_windows/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--2_audio_4197.png")
pix = np.array(pic)
print(pix)
