
from ableton_song import AbletonSong
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint

def save_and_split(path_in, midi_out, audio_out, midi_win_out='', save_midi_windows=False, n_mels=128, stepsize=30,
                   left_buffer=50, right_buffer=19, apply_sus=True, apply_denoising=False, downsample_time_dimension=False, time_dimension_factor=0.1):

    midi_bin = []
    audio_bin = []

    for filename in sorted(os.listdir(path_in)):
        if filename.endswith(".mid"):
            midi_bin.append(filename)
        elif filename.endswith(".wav"):
            audio_bin.append(filename)


    for midi_file, audio_file in zip(midi_bin, audio_bin):
        # midi_filename = midi_file[0:-5]
        # audio_filename = audio_file[0:-4]
        # assert midi_filename == audio_filename
        song = AbletonSong(os.path.join(path_in, midi_file), os.path.join(path_in, audio_file))
        try:
            song.process_audio_midi_save_slices(midi_out, audio_out, normalize_mel_spectrogram=True, n_mels=n_mels, apply_sus=apply_sus,
                                                stepsize=stepsize, left_buffer=left_buffer, right_buffer=right_buffer, apply_denoising=apply_denoising,
                                                filename='testing', file_format='png',
                                                downsample_time_dimension=downsample_time_dimension, time_dimension_factor=time_dimension_factor,
                                                save_midi_windows=save_midi_windows, midi_window_directory_path=midi_win_out)
            print('song split and saved')
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')


directory_path = '/home/zackstrater/audio_midi_repository/testforimages'
midi_out = '../data/midi_slices'
audio_out = '../data/audio_windows'
midi_windows = '../data/midi_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=25, right_buffer=10, apply_sus=False, stepsize=15,
               save_midi_windows=True, midi_win_out=midi_windows, downsample_time_dimension=True, time_dimension_factor=0.25)