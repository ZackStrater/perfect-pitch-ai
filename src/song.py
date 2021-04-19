import py_midicsv as pm
import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

class Song:
    def __init__(self, midi_filepath, audio_filepath):
        self.midi_csv_strings = pm.midi_to_csv(midi_filepath)
        self.meta_data = self.midi_csv_strings[0:7]
        self.track_end = self.midi_csv_strings[-2:]
        self.midi_data = self.midi_csv_strings[8:-2]

        header = self.meta_data[0].split(',')
        tempo = self.meta_data[2].split(',')
        ticks_per_quarter = int(header[-1])
        microsec_per_quarter = int(tempo[-1])
        microsec_per_tick = microsec_per_quarter / ticks_per_quarter
        seconds_per_tick = microsec_per_tick / 1000000
        self.tempo = microsec_per_quarter
        self.PPQ = ticks_per_quarter
        self.song_total_midi_ticks = int(self.track_end[0].split(', ')[1])
        self.midi_note_array = np.zeros((127, self.song_total_midi_ticks))
        self.midi_sus_array = np.zeros((1, self.song_total_midi_ticks))
        self.downsampled_tempo =None
        self.midi_array_splits = None

        self.audio_waveform, self.sample_rate = librosa.load(audio_filepath, sr=None)
        self.raw_spectrogram = librosa.stft(self.audio_waveform)
        self.db_spectrogram = np.flipud(librosa.amplitude_to_db(abs(self.raw_spectrogram)))
        self.song_total_sample_ticks = self.db_spectrogram.shape[0]
        self.spectrogram_array_splits = None

    '''HELPER FUNCTIONS'''
    def map_note(self, array, note_value, note_start, note_end, velocity, inverse=True):
        # maps midi notes onto the midi array
        # numpy slice isn't inclusive of note_end, as mentioned above
        # have to take abs of (127 - note_value) to make lower notes go to bottom, higher notes to top
        # note value 0 is the lowest note, needs to go to last row (127)
        # note value 127 is highest note, needs to go to first row (0)
        if inverse:
            # for midi note array
            array[np.abs(127 - note_value), note_start:note_end] = velocity
        else:
            # for sustain pedal array
            array[note_value, note_start:note_end] = velocity

    def split_into_intervals(self, array, time_interval, sample_rate, hop_length=512):
        # get sample ticks per time interval
        ticks_per_interval = time_interval * sample_rate / hop_length
        subarray_length = ticks_per_interval
        print(f'subarray length {subarray_length}')

        array_length = array.shape[1]
        print(f'array length {array_length}')
        # number of columns left over
        last_interval_length = array_length % subarray_length
        print(f'mod {last_interval_length}')
        # how many columns to add to make another full interval
        padding = subarray_length - last_interval_length
        print(f'padding {padding}')
        padded_array = np.pad(array, ((0, 0), (0, int(padding))))
        padded_array_length = padded_array.shape[1]

        num_intervals = padded_array_length / subarray_length
        print(f'num intervals {num_intervals}')
        # split array into subsections ultimately based on time_interval and sample_rate
        split_arrays = np.array_split(padded_array, num_intervals, axis=1)
        print(f'split array shape = {split_arrays[0].shape}')
        print(len(split_arrays))
        return split_arrays

    def compare_arrays(self, array, array2):
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        axs[0].imshow(array, aspect='auto')
        axs[1].imshow(array2, aspect='auto')
        plt.show()

    def apply_sus_to_slice(self, start_tick, end_tick, midi_note_array, buffer=100):
        # important to notive that end_tick is used instead of end_tick + 1
        # the end_tick is the first moment the pedal is released
        midi_slice = midi_note_array[:, start_tick: end_tick]

        def consecutive(data):
            consecutive_list = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
            if len(consecutive_list[0]) == 0:
                # if there are no consecutive nums found in a row, the above code return a blank array
                return None
            else:
                return consecutive_list

        def extend_notes_in_row(array_slice_row):
            # get indexes where there are 0's
            zeros_arg = np.argwhere(array_slice_row == 0)
            zeros_index = zeros_arg.flatten()
            # find consecutive runs of 0's
            zeros_runs = consecutive(zeros_index)  # TODO might need an if statement here
            # get start and ends of runs of 0's as list of tuples
            if zeros_runs:
                # if consecutive zeros were found:
                zeros_slices = [(arr[0], arr[-1]) for arr in zeros_runs]
                # if first slice is at the beginning, ignore it
                if zeros_slices[0][0] == 0:
                    zeros_slices.pop(0)
                for slice in zeros_slices:
                    # assign value that came directly before slice to that slice

                    '''the ':slice[1] - buffer' puts a buffer of 0's length between note and next note'''
                    array_slice_row[slice[0]: slice[1] - buffer] = array_slice_row[slice[0] - 1]
                    # if the assigned slice has the same value as the element directly following,
                    # assign the end element of the slice to 0 to differentiate the run with the following elements
                    # i.e 5, 5, 0, 0, 0, 5 . . . -> 5, 5, 5, 5, 0, 5 instead of 5, 5, 5, 5, 5, 5
                    # if slice[1] + 1 < len(array_slice_row) and array_slice_row[slice[1]] == array_slice_row[slice[1] + 1]:
                    #     array_slice_row[slice[1]] = 0
                return array_slice_row
                # return altered input row
            else:
                return array_slice_row
                # return original input row

        midi_slice_with_sus = np.apply_along_axis(extend_notes_in_row, 1, midi_slice)
        midi_note_array[:, start_tick: end_tick] = midi_slice_with_sus
        # apply to every row in array


    '''MIDI NOTE ARRAY FUNCTIONS'''
    def populate_midi_note_array(self, apply_sus=True):
        # list of midi actions to numpy array and pandas DF
        midi_values = np.genfromtxt(self.midi_data, delimiter=',', dtype=None, encoding=None)
        columns = ['track', 'tick', 'control', 'channel', 'control_num', 'velocity']
        df = pd.DataFrame(midi_values)
        df.columns = columns

        # extracting just the note presses and releases
        mask = df['control'] == ' Note_on_c'
        df_notes = df[mask]
        df_pedals = df[~mask]

        # sort notes by the note value (control_num) and tick (when they occur)
        # when velocity > 0, the note is being pressed, velocity == 0 is note being released
        df_sorted_notes = df_notes.sort_values(['control_num', 'tick']).reset_index(drop=True)
        df_key_release = df_sorted_notes[df_sorted_notes['velocity'] == 0].reset_index()
        df_key_press = df_sorted_notes[df_sorted_notes['velocity'] != 0].reset_index()

        # every note press should have a proximal note release, ensure that we have the same number of
        # presses and releases, as well as a 1:1 pairing of note presses and releases for each note
        # each row in df_key_press should be matched with a corresponding row with the same index in df_key_release
        # that specifies when that note stopped being played
        assert df_key_release.shape[0] == df_key_press.shape[0]
        assert df_key_press['control_num'].equals(df_key_release['control_num'])

        # note that 'end tick' is non inclusive
        # i.e. this is the first tick when that note stopped playing
        df_note_durations = pd.DataFrame({'start_tick': df_key_press['tick'], 'end_tick': df_key_release['tick'],
                                          'control_num': df_key_press['control_num'],
                                          'velocity': df_key_press['velocity']})

        # MIDI MATRIX W/O SUSTAIN PEDAL
        for idx, row in df_note_durations.iterrows():
            self.map_note(self.midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])

        if apply_sus:
            '''Assigning midi pedal arrays'''
            # midi maps for pedal presses
            df_sus = df_pedals[df_pedals['control_num'] == 64].reset_index(drop=True)
            # find duration by tick length to next action
            df_sus['duration'] = np.abs(df_sus['tick'].diff(periods=-1))
            # extend last action to end of song
            df_sus.loc[df_sus.index[-1], 'duration'] = self.song_total_midi_ticks - df_sus.loc[df_sus.index[-1], 'tick']

            # pedal actions record variations in how far the sustain pedal is pressed (i.e. velocity of 20 vs 80)
            # however, sustain pedal is binary, either on or off.  To get the duration where the pedal is pressed
            # only the presses directly after a release matter.  This pedal press extend until the next pedal release
            # finding releases
            sus_release_indexes = df_sus['velocity'] == 0
            # presses include the first row (first time pedal is pressed) and the rows directly after a pedal release
            sus_press_indexes = pd.Series([True, *sus_release_indexes])
            # since we added action, need to pop off last element
            sus_press_indexes = sus_press_indexes[:-1]

            df_sus_releases = df_sus[sus_release_indexes]
            df_sus_presses = df_sus[sus_press_indexes]
            assert df_sus_presses.shape[0] == df_sus_releases.shape[0]
            # MIDI tick durations where sustain pedal is pressed

            for start, end in zip(df_sus_presses['tick'], df_sus_releases['tick']):
                self.apply_sus_to_slice(start, end, self.midi_note_array)

            # midi_sus_array just for visualizing when pedal is pressed
            self.midi_sus_array = np.zeros((1, self.song_total_midi_ticks))

            # midi array for sustain pedal
            for idx, row in df_sus.iterrows():
                # mapping pedal actions to midi_sus_array
                # note_value param is 0 because midi_sus_array only has one row
                self.map_note(self.midi_sus_array, 0, row['tick'], row['tick'] + int(row['duration']), row['velocity'],
                         inverse=False)
            # pedal is either on or off, so assign all on values to 60
            self.midi_sus_array[self.midi_sus_array > 0] = 60

    def remove_top_and_bottom_midi_notes(self):
        self.midi_note_array = self.midi_note_array[10:-17, :]


    def downsample_midi_note_array(self):
        ratio = self.db_spectrogram.shape[1]/self.midi_note_array.shape[1]
        # zoom with order=0 uses nearest neighbor approach
        resized_array = zoom(self.midi_note_array, (1, ratio), order=0)
        self.midi_note_array = resized_array
        self.downsampled_tempo = self.tempo*ratio


    def remove_velocities_from_midi_note_array(self):
        self.midi_note_array[self.midi_note_array > 0] = 1

    def show_midi_note_array(self):
        plt.imshow(self.midi_note_array, aspect='auto')
        plt.show()

    '''SPECTROGRAM ARRAY FUNCTIONS'''
    # def convert_waveform_to_spectrogram(self):
    #     self.raw_spectrogram = librosa.stft(self.audio_waveform)
    #     self.db_spectrogram = librosa.amplitude_to_db(abs(self.raw_spectrogram))


    def show_spectrogram(self):
        plt.imshow(self.db_spectrogram, aspect='auto')
        plt.show()

    def remove_high_frequencies_from_spectrogram(self, frequency_ceiling=825):
        self.db_spectrogram = self.db_spectrogram[frequency_ceiling:, :]

    def apply_denoising_sigmoid(self, alpha=0.8, beta=-5):
        # db scale goes from ~-40 to 0, with 0 being the loudest
        # alpha affects the sloped of the sigmoid curve, beta centers it around a certain db value
        # alpha = 0.8, beta = -5 works well for denoising
        def db_sigmoid(x):
            return 1/(1 + np.exp(alpha*(-x-beta)))
        vectorized_db_sigmoid = np.vectorize(db_sigmoid)
        self.db_spectrogram = vectorized_db_sigmoid(self.db_spectrogram)

    '''META FUNCTIONS'''
    def split_audio_and_midi(self, time_interval):
        # split audio and downsampled  midi arrays into chunks based of length time_interval (in seconds)
        self.spectrogram_array_splits = self.split_into_intervals(self.db_spectrogram, time_interval, self.sample_rate)
        self.midi_note_array_splits = self.split_into_intervals(self.midi_note_array, time_interval, self.sample_rate)
        assert len(self.spectrogram_array_splits) == len(self.midi_note_array_splits)

    def save_splits(self, midi_directory_path, audio_directory_path, filename=''):
        for i, (midi, audio) in enumerate(zip(self.midi_note_array_splits, self.spectrogram_array_splits)):
            with open(f'{midi_directory_path}/{filename}_midi_{i}.npy', 'wb') as f:
                np.save(f, midi)
            with open(f'{audio_directory_path}/{filename}_audio_{i}.npy', 'wb') as f:
                np.save(f, audio)

    def format_split_save_synced_midi_audio_files(self, midi_directory_path, audio_directory_path, filename='', time_interval=8, spectrogram_freq_ceiling=825,
                                                  remove_high_frequencies=True, remove_velocity=True, remove_top_and_bottom_midi_notes=True,
                                                  downsample=True, apply_denoising=False, alpha=0.8, beta=-5, ):
        self.populate_midi_note_array()
        if remove_high_frequencies:
            self.remove_high_frequencies_from_spectrogram(frequency_ceiling=spectrogram_freq_ceiling)
        if remove_velocity:
            self.remove_velocities_from_midi_note_array()
        if remove_top_and_bottom_midi_notes:
            self.remove_top_and_bottom_midi_notes()
        if apply_denoising:
            self.apply_denoising_sigmoid(alpha, beta)
        if downsample:
            self.downsample_midi_note_array()
        self.split_audio_and_midi(time_interval)
        self.save_splits(midi_directory_path, audio_directory_path, filename=filename)


if __name__ == '__main__':

    new_song = Song('../data/test_midi_csv.midi', '../data/test_aduio.wav')

    midi_filepath = '/home/zackstrater/Desktop/test_midi_audio_files/midi_files'
    audio_filepath = '/home/zackstrater/Desktop/test_midi_audio_files/audio_files'
    new_song.format_split_save_synced_midi_audio_files(midi_filepath, audio_filepath, filename='testing')

    import os
    # midi_path = '/home/zackstrater/Desktop/test_midi_audio_files/midi_files'
    # for filename in sorted(os.listdir(midi_path)):
    #     print(filename)
    #     array = np.load(os.path.join(midi_path, filename))
    #     print(array)
    #     print(array.shape)
    #     new_song.compare_arrays(array, array)

    audio_path = '/home/zackstrater/Desktop/test_midi_audio_files/audio_files'
    for filename in sorted(os.listdir(audio_path)):
        print(filename)
        array = np.load(os.path.join(audio_path, filename))
        print(array)
        print(array.shape)
        new_song.compare_arrays(array, array)