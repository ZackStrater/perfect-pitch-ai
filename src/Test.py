


from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys

midi_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi'
audio_filepath = '/media/zackstrater/New Volume/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav'
new_song = Song(midi_filepath, audio_filepath)
new_song.format_split_save_synced_midi_audio_files('../data', '../data', filename='fixing_sus')

