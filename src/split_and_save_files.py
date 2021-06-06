
from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint





def save_and_split(path_in, midi_out, audio_out, midi_win_out='', save_midi_windows=False, n_mels=200, stepsize=20,
                   left_buffer=10, right_buffer=9, apply_sus=False, apply_denoising=False,
                   downsample_time_dimension=False, time_dimension_factor=0.5, sample_rate=None, CQT=False, VQT=False):

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
        song = Song(os.path.join(path_in, midi_file), os.path.join(path_in, audio_file), sample_rate=sample_rate)
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



# complete database processing

midi_out = '/home/zackstrater/audio_midi_repository/128mel_50L_49R_0,5ds_NOsus_step100/midi_slices'
audio_out = '/home/zackstrater/audio_midi_repository/128mel_50L_49R_0,5ds_NOsus_step100/audio_windows'
midi_win_out = '/home/zackstrater/audio_midi_repository/128mel_50L_49R_0,5ds_NOsus_step100/midi_windows'

directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2018'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)

print('2018 done')

directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2017'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)

print('2017 done')


directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2015'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)
print('2015 done')


directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2014'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)


print('2014 done')


# directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2013'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# print('2013 done')


directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2011'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)

print('2011 done')

directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2009'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)


print('2009 done')

# directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2008'
#
# save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=10, apply_sus=True, stepsize=61,
#                save_midi_windows=False, midi_win_out='', downsample_time_dimension=True, time_dimension_factor=0.25)
#
# print('2008 done')

directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2006'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)


print('2006 done')

directory_path = '/media/zackstrater/Samsung_T5/maestro-v3.0.0/2004'

save_and_split(directory_path, midi_out, audio_out, n_mels=128, left_buffer=50, right_buffer=49, apply_sus=False, stepsize=100,
               save_midi_windows=True, midi_win_out=midi_win_out, downsample_time_dimension=True, time_dimension_factor=0.5)


print('2004 done')




