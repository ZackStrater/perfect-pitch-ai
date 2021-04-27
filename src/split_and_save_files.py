
from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint

def save_and_split(path_in, midi_out, audio_out, midi_win_out='', save_midi_windows=False, n_mels=128, stepsize=30,
                   left_buffer=50, right_buffer=19, apply_sus=True, apply_denoising=False,
                   downsample_time_dimension=False, time_dimension_factor=0.1, CQT=False, VQT=False):

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
                                                downsample_time_dimension=downsample_time_dimension,
                                                time_dimension_factor=time_dimension_factor,
                                                save_midi_windows=save_midi_windows, midi_window_directory_path=midi_win_out,
                                                CQT=CQT, VQT=VQT)
            print('song split and saved')
        except AssertionError:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')




# midi_out = '/home/zackstrater/audio_midi_repository/sus_200mels_50l_20r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_200mels_50l_20r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=200, left_buffer=50, right_buffer=19)
#
# midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_10l_10r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_10l_10r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=10, right_buffer=10)
#
# midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_50r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_50l_50r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=50)
#
#
# midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_100r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_100r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=100, right_buffer=100)
#
# midi_out = '/home/zackstrater/audio_midi_repository/no_sus_128mels_50l_20r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/no_sus_128mels_50l_20r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_sus=False)
#
# midi_out = '/home/zackstrater/audio_midi_repository/denoise_sus_128mels_50l_20r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/denoise_sus_128mels_50l_20r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_denoising=True)
#
# midi_out = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_20r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_20r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=19, apply_sus=False, apply_denoising=True)


# midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_9l_90r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_9l_90r_step30/audio_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=9, right_buffer=90, apply_sus=True)
#
#
# midi_out = '/home/zackstrater/audio_midi_repository/sus_128mels_90l_9r_step30/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/sus_128mels_90l_9r_step30/audio_windows'
# midi_win_out = '/home/zackstrater/audio_midi_repository/sus_128mels_90l_9r_step30/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=90, right_buffer=9, apply_sus=True, save_midi_windows=True, midi_win_out=midi_win_out)


# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,05_sus_128mels_3l_1r_step_2_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,05_sus_128mels_3l_1r_step_2_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=3, right_buffer=1, apply_sus=True, stepsize=2,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.05)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_100l_5r_step_3_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_100l_5r_step_3_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=100, right_buffer=5, apply_sus=True, stepsize=3,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.1)
#
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_50l_20r_step_3_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_50l_20r_step_3_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=20, apply_sus=True, stepsize=3,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.1)

# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_50l_40r_step_3_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,1_sus_128mels_50l_40r_step_3_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=40, apply_sus=True, stepsize=3,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.1)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,05_sus_200mels_50l_10r_step_2_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,05_sus_200mels_50l_10r_step_2_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=200, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=2,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.05)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,025_sus_128mels_50l_20r_step_3_zoom1/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,025_sus_128mels_50l_20r_step_3_zoom1/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=20, apply_sus=True, stepsize=1,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.025)


# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_10l_10r_step_60/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_10l_10r_step_60/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=10, right_buffer=10, apply_sus=True, stepsize=60,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_25l_10r_step_60/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_25l_10r_step_60/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=25, right_buffer=10, apply_sus=True, stepsize=60,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_50l_10r_step_60/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_50l_10r_step_60/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=60,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)


# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_10l_10r_step_21/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_10l_10r_step_21/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=10, right_buffer=10, apply_sus=True, stepsize=21,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_50l_20r_step_61/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_128mels_50l_20r_step_61/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=20, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# midi_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_200mels_50l_10r_step_61/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/downsample0,25_sus_200mels_50l_10r_step_61/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=200, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)

# directory_path = '/home/zackstrater/audio_midi_repository/2017'
# midi_out = '/home/zackstrater/audio_midi_repository/2017downsample0,25_sus_128mels_50l_10r_step_61/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/2017downsample0,25_sus_128mels_50l_10r_step_61/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
#
# directory_path = '/home/zackstrater/audio_midi_repository/2018'
#
# midi_out = '/home/zackstrater/audio_midi_repository/2018downsample0,25_sus_128mels_50l_10r_step_61/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/2018downsample0,25_sus_128mels_50l_10r_step_61/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# midi_out = '/home/zackstrater/audio_midi_repository/2018downsample0,5_sus_128mels_100l_20r_step_121/midi_slices'
# audio_out = '/home/zackstrater/audio_midi_repository/2018downsample0,5_sus_128mels_100l_20r_step_121/audio_windows'
# # midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=100, right_buffer=20, apply_sus=True, stepsize=121,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.5)


directory_path = '/home/zackstrater/audio_midi_repository/2018'

midi_out = '/home/zackstrater/audio_midi_repository/2018CQT/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/2018CQT/audio_windows'
# midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
               save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25, CQT=True)

midi_out = '/home/zackstrater/audio_midi_repository/2018VQT/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/2018VQT/audio_windows'
# midi_win_out = '/home/zackstrater/audio_midi_repository/downsample0,5_25l_10r_step15/midi_windows'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
               save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25, VQT=True)