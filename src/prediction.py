


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
from tensorflow import keras
import tensorflow
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)


midi_filepath = '/home/zackstrater/audio_midi_repository/predictions/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'
audio_filepath = '/home/zackstrater/audio_midi_repository/testing/Cscaleharmony.wav'
midi_out=''
audio_out=''
new_song = Song(midi_filepath, audio_filepath)
new_song.process_audio_midi_save_slices(midi_out, audio_out, normalize_mel_spectrogram=True, n_mels=200, apply_sus=False,
                                                stepsize=20, left_buffer=10, right_buffer=9, apply_denoising=False,
                                                filename='', file_format='png',
                                                downsample_time_dimension=True,
                                                time_dimension_factor=0.5, save=False,
                                                save_midi_windows=False, midi_window_directory_path='')




model = keras.models.load_model('/home/zackstrater/DSIclass/capstones/audio-to-midi/models/200mel_10L_9R_0,5ds_NOsus_step20_epoch6')
new_song.make_predictions(model, 10, 9)
new_song.export_midi_prediction_array('../data/harmony_cscale.midi')

