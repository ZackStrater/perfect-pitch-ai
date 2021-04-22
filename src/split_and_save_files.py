
from song import Song
import py_midicsv as pm
import librosa.display
import os
import traceback
import sys
from termcolor import cprint

midi_bin = []
audio_bin = []
directory_path = '/home/zackstrater/audio_midi_repository/2018'
midi_out_path = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_27r_step20/midi_slices'
audio_out_path = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_27r_step20/audio_windows'
midi_win_out_path = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_27r_step20/midi_windows'

# sus 128 mels 100l 27 right
# midi_out_path = '../data/midi_slices'
# audio_out_path = '../data/audio_windows'
# midi_win_out_path = '../data/midi_windows'


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
        song.process_audio_midi_save_slices(midi_out_path, audio_out_path, normalize_mel_spectrogram=True, n_mels=128,
                                            stepsize=20, left_buffer=100, right_buffer=27, filename=midi_filename, file_format='png',
                                            save_midi_windows=True, midi_window_directory_path=midi_win_out_path)
        print('song split and saved')
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        cprint('An error occurred on line {} in statement {}'.format(line, text), 'red')


