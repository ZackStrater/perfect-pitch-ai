import py_midicsv as pm
import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from PIL import Image
import warnings

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
        self.midi_note_array = np.zeros((128, self.song_total_midi_ticks))
        self.midi_sus_array = np.zeros((1, self.song_total_midi_ticks))
        self.downsampled_midi_ratio = 1
        self.midi_array_splits = None

        self.audio_waveform, self.sample_rate = librosa.load(audio_filepath, sr=None)
        self.raw_spectrogram = librosa.stft(self.audio_waveform)
        self.db_spectrogram = np.flipud(librosa.amplitude_to_db(abs(self.raw_spectrogram)))
        self.song_total_sample_ticks = self.db_spectrogram.shape[0]
        self.spectrogram_array_splits = None

        self.mel_spectrogram = None
        self.mel_windows = None
        self.midi_windows = None
        self.midi_slices = None

        self.prediction_audio_array = None
        self.prediction_audio_windows = None
        self.prediction_midi_slices = None
        self.prediction_midi_array = None

        self.export_midi_array = None

    '''HELPER FUNCTIONS'''
    def map_note(self, array, note_value, note_start, note_end, velocity, inverse=True):
        # maps midi notes onto the midi array
        # numpy slice isn't inclusive of note_end, as mentioned above
        # have to take abs of (128 - note_value) to make lower notes go to bottom, higher notes to top
        # note value 0 is the lowest note, needs to go to last row (128)
        # note value 128 is highest note, needs to go to first row (0)
        if inverse:
            # for midi note array
            array[np.abs(128 - note_value), note_start:note_end] = velocity
        else:
            # for sustain pedal array
            array[note_value, note_start:note_end] = velocity

    def get_window_indices(self, array, stepsize, left_buffer, right_buffer):
        array_len = array.shape[1]
        first_sample = left_buffer
        last_sample = array_len - right_buffer
        center_indices = np.arange(first_sample, last_sample, stepsize)

        def left_right_indices(center, left, right):
            return center - left, center + right + 1
        vec_left_right_indices = np.vectorize(left_right_indices)
        left_indices, right_indices = vec_left_right_indices(center_indices, left_buffer, right_buffer)
        return left_indices, right_indices, center_indices

    def get_windows(self, array, left_indicies, right_indicies):
        window_bin = []
        for l, r in zip(left_indicies, right_indicies):
            window_bin.append(array[:, l:r])
        return window_bin

    def get_midi_slices(self, array, center_indices):
        midi_slice_bin = []
        for c in center_indices:
            midi_slice_bin.append(array[:, c])
        return midi_slice_bin

    def split_into_intervals(self, array, time_interval, sample_rate, hop_length=512):
        # get sample ticks per time interval
        ticks_per_interval = time_interval * sample_rate / hop_length
        subarray_length = ticks_per_interval

        array_length = array.shape[1]
        # number of columns left over
        last_interval_length = array_length % subarray_length
        # how many columns to add to make another full interval
        padding = subarray_length - last_interval_length
        padded_array = np.pad(array, ((0, 0), (0, int(padding))))
        padded_array_length = padded_array.shape[1]

        num_intervals = padded_array_length / subarray_length
        # split array into subsections ultimately based on time_interval and sample_rate
        split_arrays = np.array_split(padded_array, num_intervals, axis=1)
        return split_arrays

    def compare_arrays(self, *arrays):
        num_arrays = len(arrays)
        fig, axs = plt.subplots(1, num_arrays, figsize=(5*num_arrays, 20))
        for i in range(num_arrays):
            print(arrays[i].shape)
            axs[i].imshow(arrays[i], aspect='auto', interpolation='nearest')
        plt.show()

    def apply_sus_to_slice(self, start_tick, end_tick, midi_note_array, buffer=200):
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
            # extend last action (usually releasing sus_pedal) to end of song
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

            if df_sus_presses.shape[0] != df_sus_releases.shape[0]:
                df_sus_releases = df_sus_releases.append({'track': 2, 'tick': self.song_total_midi_ticks, 'control': 'Control_c',
                                        'channel':0, 'control_num': 64, 'velocity': 0, 'duration': 0}, ignore_index=True)

                print('sustain pedal issue')
                print(df_sus_presses.shape[0])
                print(df_sus_releases.shape[0])
                print(df_sus_releases)
                print(df_sus_presses)
                print(self.song_total_midi_ticks)
            assert df_sus_presses.shape[0] == df_sus_releases.shape[0], 'assertion error sustain'
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

    def convert_midi_array_to_pianoroll(self):
        self.midi_note_array = self.midi_note_array[21:109, :]

    def downsample_midi_note_array(self):
        ratio = self.db_spectrogram.shape[1]/self.midi_note_array.shape[1]
        # zoom with order=0 uses nearest neighbor approach
        resized_array = zoom(self.midi_note_array, (1, ratio), order=0)
        self.midi_note_array = resized_array
        self.downsampled_midi_ratio *= ratio


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

    def show_mel_spectrogram(self):
        plt.imshow(self.mel_spectrogram, aspect='auto')
        plt.show()

    def remove_high_frequencies_from_spectrogram(self, frequency_ceiling=825):
        self.db_spectrogram = self.db_spectrogram[frequency_ceiling:, :]

    def normalize_mel_spectrogram(self):
        self.mel_spectrogram = self.mel_spectrogram/np.abs(np.min(self.mel_spectrogram)) + 1

    def apply_denoising_sigmoid(self, alpha=8, beta=4):
        # gets rid of noise and intensifies existing signals
        # alpha determines the 'slope' of the sigmoid curve (higher alpha = steeper)
        # beta moves sigmoid left to right
        # alpha = 8, beta = 4 is well calibrated for a normalized mel spectrum (values ranging from 0 to 1)
        # values < 0.5 are reduced, values > 0.5 are amplified
        # more accurate: 1.02/(1 + np.exp(-8*x-4)) - 0.0183
        def db_sigmoid(x):
            return 1/(1 + np.exp(-alpha*x+beta))
        vectorized_db_sigmoid = np.vectorize(db_sigmoid)
        self.mel_spectrogram = vectorized_db_sigmoid(self.mel_spectrogram)

    def downsample_time_dimension(self, factor=0.25):
        # zoom with order=0 uses nearest neighbor approach
        resized_midi_array = zoom(self.midi_note_array, (1, factor), order=0)
        self.midi_note_array = resized_midi_array
        resized_audio_array = zoom(self.mel_spectrogram, (1, factor), order=1)
        self.mel_spectrogram = resized_audio_array
        self.downsampled_midi_ratio *= factor

    '''META FUNCTIONS'''
    # def split_audio_and_midi_into_equal_partitions(self, time_interval):
    #     # split audio and downsampled  midi arrays into chunks based of length time_interval (in seconds)
    #     self.spectrogram_array_splits = self.split_into_intervals(self.db_spectrogram, time_interval, self.sample_rate)
    #     self.midi_note_array_splits = self.split_into_intervals(self.midi_note_array, time_interval, self.sample_rate)
    #     assert len(self.spectrogram_array_splits) == len(self.midi_note_array_splits)
    #
    # def save_splits(self, midi_directory_path, audio_directory_path, filename=''):
    #     for i, (midi, audio) in enumerate(zip(self.midi_note_array_splits, self.spectrogram_array_splits)):
    #         with open(f'{midi_directory_path}/{filename}_midi_{i}.npy', 'wb') as f:
    #             np.save(f, midi)
    #         with open(f'{audio_directory_path}/{filename}_audio_{i}.npy', 'wb') as f:
    #             np.save(f, audio)
    #
    # def format_split_save_synced_midi_audio_files(self, midi_directory_path, audio_directory_path, filename='', time_interval=8, spectrogram_freq_ceiling=825,
    #                                               apply_sus=True, remove_high_frequencies=True, remove_velocity=True, convert_midi_to_pianoroll=True,
    #                                               downsample=True, apply_denoising=False, alpha=8, beta=4):
    #     if apply_sus:
    #         self.populate_midi_note_array()
    #     else:
    #         self.populate_midi_note_array(apply_sus=False)
    #     if remove_high_frequencies:
    #         self.remove_high_frequencies_from_spectrogram(frequency_ceiling=spectrogram_freq_ceiling)
    #     if remove_velocity:
    #         self.remove_velocities_from_midi_note_array()
    #     if convert_midi_to_pianoroll:
    #         self.convert_midi_array_to_pianoroll()
    #     if apply_denoising:
    #         self.apply_denoising_sigmoid(alpha, beta)
    #     if downsample:
    #         self.downsample_midi_note_array()
    #     self.split_audio_and_midi_into_equal_partitions(time_interval)
    #     self.save_splits(midi_directory_path, audio_directory_path, filename=filename)


    '''MEL SPECTROGRAM FORMATTING AND PROCESSING'''
    def process_mel_spectrogram(self, n_mels, CQT=False, VQT=False):
        if CQT:
            C = np.abs(librosa.cqt(self.audio_waveform, sr=self.sample_rate, n_bins=112))
            CQT = librosa.amplitude_to_db(C, ref=np.max)
            self.mel_spectrogram = np.flipud(CQT)
        elif VQT:
            V = np.abs(librosa.vqt(self.audio_waveform, sr=self.sample_rate, n_bins=112))
            VQT = librosa.amplitude_to_db(V, ref=np.max)
            self.mel_spectrogram = np.flipud(VQT)
        else:
            mel_spectrogram = np.flipud(librosa.feature.melspectrogram(self.audio_waveform, sr=self.sample_rate, n_mels=n_mels))
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            self.mel_spectrogram = log_mel_spectrogram

    def get_audio_windows_and_midi_slices(self, audio_array, midi_array, stepsize, left_buffer, right_buffer):
        left_indices, right_indices, center_indices = self.get_window_indices(audio_array, stepsize, left_buffer, right_buffer)
        audio_windows = self.get_windows(audio_array, left_indices, right_indices)
        midi_windows = self.get_windows(midi_array, left_indices, right_indices)
        midi_slices = self.get_midi_slices(midi_array, center_indices)
        return audio_windows, midi_slices, midi_windows

    def save_audio_windows_midi_splits(self, midi_directory_path, audio_directory_path, filename='', file_format='png', save_midi_windows=False, midi_window_directory_path=''):
        if file_format not in ['npy', 'png', 'bmp', 'jpeg']:
            warnings.warn('WARNING: Unsupported filetype')
        if save_midi_windows:
            for i, (midi, audio, midi_win) in enumerate(zip(self.midi_slices, self.mel_windows, self.midi_windows)):
                with open(f'{midi_directory_path}/{filename}_midi_{i}.npy', 'wb') as f:
                    np.save(f, midi)

                with open(f'{audio_directory_path}/{filename}_audio_{i}.{file_format}', 'wb') as f:
                    im = Image.fromarray((audio*255).astype(np.uint8))
                    im.save(f)

                with open(f'{midi_window_directory_path}/{filename}_mwin_{i}.{file_format}', 'wb') as f:
                    im = Image.fromarray((midi_win*255).astype(np.uint8))
                    im.save(f)


        else:
            for i, (midi, audio) in enumerate(zip(self.midi_slices, self.mel_windows)):
                with open(f'{midi_directory_path}/{filename}_midi_{i}.npy', 'wb') as f:
                    np.save(f, midi)
                with open(f'{audio_directory_path}/{filename}_audio_{i}.{file_format}', 'wb') as f:
                    im = Image.fromarray((audio*255).astype(np.uint8))
                    im.save(f)

    # def save_audio_windows_midi_splits(self, midi_directory_path, audio_directory_path, filename='', save_midi_windows=False, midi_window_directory_path=''):
    #     if save_midi_windows:
    #         with open(f'{midi_directory_path}/{filename}_midi.npy', 'wb') as f:
    #             print(np.array(self.midi_slices).shape)
    #             np.save(f, np.array(self.midi_slices))
    #             print('saved midi slice')
    #         with open(f'{audio_directory_path}/{filename}_audio.npy', 'wb') as f:
    #             print(np.array(self.mel_windows).shape)
    #             np.save(f, np.array(self.mel_windows))
    #             print('saved audio window')
    #
    #         with open(f'{midi_window_directory_path}/{filename}_mwin.npy', 'wb') as f:
    #             print(np.array(self.midi_windows).shape)
    #             np.save(f, np.array(self.midi_windows))
    #             print('saved midi_windows')
    #
    #     else:
    #         with open(f'{midi_directory_path}/{filename}_midi.npy', 'wb') as f:
    #             np.save(f, np.array(self.midi_slices))
    #         with open(f'{audio_directory_path}/{filename}_audio.npy', 'wb') as f:
    #             np.save(f, np.array(self.mel_windows))


    def process_audio_midi_save_slices(self,
                                       midi_directory_path, audio_directory_path, # path info
                                       n_mels, stepsize, left_buffer, right_buffer, # audio info
                                       normalize_mel_spectrogram=True, apply_denoising=False, alpha=8, beta=4,
                                       apply_sus=True, remove_velocity=True, # midi info
                                       convert_midi_to_pianoroll=True, downsample = True,
                                       downsample_time_dimension=False, time_dimension_factor=0.1,
                                       filename='', file_format='png', save=True, save_midi_windows=False, midi_window_directory_path='',
                                       CQT=False, VQT=False):
        # MIDI FUNCS
        if apply_sus:
            self.populate_midi_note_array()
        else:
            self.populate_midi_note_array(apply_sus=False)
        if remove_velocity:
            self.remove_velocities_from_midi_note_array()
        if convert_midi_to_pianoroll:
            self.convert_midi_array_to_pianoroll()
        if downsample:
            self.downsample_midi_note_array()

        # AUDIO FUNCS
        self.process_mel_spectrogram(n_mels, CQT=CQT, VQT=VQT)
        if normalize_mel_spectrogram:
            self.normalize_mel_spectrogram()
        if apply_denoising:
            self.apply_denoising_sigmoid(alpha, beta)

        # DOWNSAMPLING MIDI AND AUDIO
        if downsample_time_dimension:
            self.downsample_time_dimension(time_dimension_factor)

        # SPLIT SONG
        self.mel_windows, self.midi_slices, self.midi_windows = self.get_audio_windows_and_midi_slices(self.mel_spectrogram, self.midi_note_array, stepsize, left_buffer, right_buffer)
        if save_midi_windows and save:
            self.save_audio_windows_midi_splits(midi_directory_path, audio_directory_path, filename=filename,
                                                file_format=file_format, save_midi_windows=True,
                                                midi_window_directory_path=midi_window_directory_path)
        elif save:
            self.save_audio_windows_midi_splits(midi_directory_path, audio_directory_path, filename=filename,
                                                file_format=file_format)


    def make_predictions(self, model, left_buffer, right_buffer):
        self.prediction_audio_array = np.pad(self.mel_spectrogram, ((0, 0), (left_buffer, right_buffer)))
        left_indicies, right_indices, center_indices = self.get_window_indices(self.prediction_audio_array, 1, left_buffer, right_buffer)
        self.prediction_audio_windows = np.array(self.get_windows(self.prediction_audio_array, left_indicies, right_indices))
        imgs = self.prediction_audio_windows.shape[0]
        rows = self.prediction_audio_windows.shape[1]
        columns = self.prediction_audio_windows.shape[2]
        self.prediction_audio_windows = self.prediction_audio_windows.reshape((imgs, rows, columns, 1))

        self.prediction_midi_slices = model.predict(self.prediction_audio_windows)
        self.prediction_midi_array = np.column_stack(self.prediction_midi_slices)
        # # np.set_printoptions(threshold=np.inf)
        # # print(self.prediction_midi_slices)
        # np.set_printoptions(threshold=np.inf)
        # np.set_printoptions(linewidth=200)
        # np.set_printoptions(suppress=True)
        # print(self.mel_spectrogram[:, 400:450])
        # # print(self.prediction_midi_array[:, 400:450])

        self.prediction_midi_array[self.prediction_midi_array < 0.5] = 0
        self.prediction_midi_array[self.prediction_midi_array >= 0.5] = 1
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(self.midi_note_array, aspect='auto', interpolation='nearest')
        # axs[1].imshow(self.prediction_midi_array, aspect='auto', interpolation='nearest')
        # axs[2].imshow(self.mel_spectrogram, aspect='auto', interpolation='nearest')
        # plt.show()

    def export_midi_prediction_array(self, midi_array, path_out):
        self.export_midi_array = np.flipud(np.pad(midi_array, ((20,20),(0,0))))
        self.export_midi_array[self.export_midi_array > 0] = 60

        def find_runs(x, row_index):
            # finds continuous segments of note presses and returns the value, start, length, and velocity for each note

            n = x.shape[0]
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True

            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # run_velocity -> take the value of first element in the run
            run_velocities = x[loc_run_start]

            # find run lengths, unused
            run_lengths = np.diff(np.append(run_starts, n))

            # need to make a np array with note value (passed in as row_index) that is the same length as
            # the number of notes found for that row
            note_values = np.full(len(run_starts), row_index)

            # return array where each row has [note value, start of run, velocity of note]
            # note value is what note was played, determined by row_index passed in
            return np.dstack((note_values, run_starts, run_velocities))[0]

        # gathering note presses for each row (note) in midi array in order
        note_presses_and_releases = np.vstack([find_runs(row, idx) for idx, row in enumerate(self.export_midi_array)])

        # remove actions where the start tick and velocity are both 0
        # these are runs of 0's found at the beginning of the track for each note
        # if counted, it would include these as a note release for each note at the beginning of track, so they are excluded
        mask = (note_presses_and_releases[:, 1] == 0) & (note_presses_and_releases[:, 2] == 0)
        note_presses_and_releases = note_presses_and_releases[~mask]

        # IF YOU WANT TO COMBINE SUS PEDAL AND NOTE PRESSES:
        # sus_pedal_actions = np.vstack(find_runs(midi_sus_array[0], 128))
        # mask = (sus_pedal_actions[:, 1] == 0) & (sus_pedal_actions[:, 2] == 0)
        # sus_pedal_actions = sus_pedal_actions[~mask]
        # all_midi_actions = np.vstack([note_presses_and_releases, sus_pedal_actions])
        # sorted_all_midi_actions = all_midi_actions[all_midi_actions[:, 1].argsort()]

        # JUST NOTE PRESSES AND RELEASES:
        sorted_note_presses_and_releases = note_presses_and_releases[note_presses_and_releases[:, 1].argsort()]

        def write_midi_line(track, tick, control, channel, control_num, velocity):
            midi_string = ', '.join(
                [str(track), str(tick), str(control), str(channel), str(control_num), str(velocity)])
            midi_string += '\n'
            return midi_string

        # recombining midi actions with metadata and end of file strings
        midi_out = []
        for line in self.meta_data:
            if 'Tempo' in line:
                new_line = f'1, 0, Tempo, {round(self.tempo/self.downsampled_midi_ratio)}\n'
                midi_out.append(new_line)
            else:
                midi_out.append(line)
        for line in sorted_note_presses_and_releases:
                midi_out.append(write_midi_line(2, int(line[1]), 'Note_on_c', 0, int(line[0]), int(line[2])))
        for line in self.track_end:
            if 'End_track' in line:
                new_line = f'2, {round(self.song_total_midi_ticks*self.downsampled_midi_ratio)}, End_track\n'
                midi_out.append(new_line)
            else:
                midi_out.append(line)
        midi_object = pm.csv_to_midi(midi_out)
        with open(path_out, 'wb') as output_file:
            midi_writer = pm.FileWriter(output_file)
            midi_writer.write(midi_object)

