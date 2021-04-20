
from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint

midi_bin = []
audio_bin = []
directory_path = '/media/zackstrater/New Volume/maestro-v3.0.0/2017'
midi_out_path = '/media/zackstrater/New Volume/split_audio_midi_files/midi_files'
audio_out_path = '/media/zackstrater/New Volume/split_audio_midi_files/audio_files'

for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".midi"):
        midi_bin.append(filename)
    elif filename.endswith(".wav"):
        audio_bin.append(filename)


for midi_file, audio_file in zip(midi_bin, audio_bin):
    midi_filename = midi_file[0:-5]
    audio_filename = audio_file[0:-4]
    assert midi_filename == audio_filename
    song = Song(os.path.join(directory_path, midi_file), os.path.join(directory_path, audio_file))
    try:
        song.format_split_save_synced_midi_audio_files(midi_out_path, audio_out_path, filename=midi_filename)
        print('song split and saved')
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')

