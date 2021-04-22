


from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
import numpy as np

midi_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--1.midi'
audio_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--1.wav'
new_song = Song(midi_filepath, audio_filepath)
new_song.process_audio_midi_save_slices('../data', '../data', normalize_mel_spectrogram=True, apply_sus=False, apply_denoising=False, alpha=8, beta=4, n_mels=128, stepsize=20, left_buffer=100, right_buffer=27, filename='44100hz', save=False)
for midi_win, midi, audio in zip(new_song.midi_windows, new_song.midi_slices, new_song.mel_windows):
    new_song.compare_arrays(midi_win, np.tile(midi, (10, 1)).T, audio)

