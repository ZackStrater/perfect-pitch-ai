


from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
import numpy as np

midi_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi'
audio_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.wav'
new_song = Song(midi_filepath, audio_filepath)
new_song.process_audio_midi_save_slices('../data', '../data', n_mels=128, stepsize=10, left_buffer=100, right_buffer=10, filename='44100hz')
for midi_win, midi, audio in zip(new_song.midi_windows, new_song.midi_slices, new_song.mel_windows):
    new_song.compare_arrays(np.tile(midi, (10, 1)).T, audio)

