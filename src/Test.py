


from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys

midi_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi'
audio_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.wav'
new_song = Song(midi_filepath, audio_filepath)
new_song.format_split_save_synced_midi_audio_files('../data', '../data', filename='44100hz')
for midi, audio in zip(new_song.midi_note_array_splits, new_song.spectrogram_array_splits):
    new_song.compare_arrays(midi, audio)

