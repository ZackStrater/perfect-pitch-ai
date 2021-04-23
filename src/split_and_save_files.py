
from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint

def save_and_split(path_in, midi_out, audio_out, midi_win_out='', save_midi_windows=False, n_mels=128, stepsize=30,
                   left_buffer=50, right_buffer=19, apply_sus=True, apply_denoising=False):

    midi_bin = []
    audio_bin = []

    for filename in sorted(os.listdir(path_in)):
        if filename.endswith(".midi"):
            midi_bin.append(filename)
        elif filename.endswith(".wav"):
            audio_bin.append(filename)


    for midi_file, audio_file in zip(midi_bin, audio_bin):
        midi_filename = midi_file[0:-5]
        audio_filename = audio_file[0:-4]
        assert midi_filename == audio_filename
        song = Song(os.path.join(path_in, midi_file), os.path.join(path_in, audio_file))
        try:
            song.process_audio_midi_save_slices(midi_out, audio_out, normalize_mel_spectrogram=True, n_mels=n_mels, apply_sus=apply_sus,
                                                stepsize=stepsize, left_buffer=left_buffer, right_buffer=right_buffer, apply_denoising=apply_denoising,
                                                filename=midi_filename, file_format='png',
                                                save_midi_windows=save_midi_windows, midi_window_directory_path=midi_win_out)
            print('song split and saved')
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')

# def save_and_split(path_in, midi_out, audio_out, midi_win_out='', save_midi_windows=False, n_mels=128, stepsize=30,
#                    left_buffer=50, right_buffer=19, apply_sus=True, apply_denoising=False):
directory_path = '/home/zackstrater/audio_midi_repository/2018'


midi_out = '/home/zackstrater/audio_midi_repository/sus_200mels_50l_20r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/sus_200mels_50l_20r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=200, left_buffer=50, right_buffer=19)

midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_10l_10r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_10l_10r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=10, right_buffer=10)

midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_50r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_50r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=50)


midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_100r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_100r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=100, right_buffer=100)

midi_out = '/home/zackstrater/audio_midi_repository/no_sus_128mels_50l_20r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/no_sus_128mels_50l_20r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_sus=False)

midi_out = '/home/zackstrater/audio_midi_repository/denoise_sus_128mels_50l_20r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/denoise_sus_128mels_50l_20r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_denoising=True)

midi_out = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_20r_step30/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_20r_step30/audio_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_sus=False, apply_denoising=True)












# for filename in sorted(os.listdir(directory_path)):
#     if filename.endswith(".midi"):
#         midi_bin.append(filename)
#     elif filename.endswith(".wav"):
#         audio_bin.append(filename)
#
#
# for midi_file, audio_file in zip(midi_bin, audio_bin):
#     midi_filename = midi_file[0:-5]
#     audio_filename = audio_file[0:-4]
#     assert midi_filename == audio_filename
#     song = Song(os.path.join(directory_path, midi_file), os.path.join(directory_path, audio_file))
#     try:
#         song.process_audio_midi_save_slices(midi_out_path, audio_out_path, normalize_mel_spectrogram=True, n_mels=128,
#                                             stepsize=30, left_buffer=50, right_buffer=19, filename=midi_filename, file_format='png',
#                                             save_midi_windows=True, midi_window_directory_path=midi_win_out_path)
#         print('song split and saved')
#     except AssertionError:
#         _, _, tb = sys.exc_info()
#         tb_info = traceback.extract_tb(tb)
#         filename, line, func, text = tb_info[-1]
#
#         cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')