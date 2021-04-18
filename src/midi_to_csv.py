import py_midicsv as pm
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from scipy.ndimage import zoom



# import midi file, result is a list of strings that contain midi actions
csv_string = pm.midi_to_csv('../data/test_midi_csv.midi')

# separate metadata and endfile data, midi_data contains all the information about note presses and releases
meta_data = csv_string[0:7]
track_end = csv_string[-2:]
# for line in meta_data:
#     print(line)
midi_data = csv_string[8:-2]

# list of midi actions to numpy array and pandas DF
midi_array = np.genfromtxt(midi_data, delimiter=',', dtype=None, encoding=None)
columns = ['track', 'tick', 'control', 'channel', 'control_num', 'velocity']
df = pd.DataFrame(midi_array)
df.columns = columns

# getting time from PPQ (pulses per quarternote) in header and tempo (usually 500000 microseconds per beat)
header = meta_data[0].split(',')
tempo = meta_data[2].split(',')
ticks_per_quarter = int(header[-1])
microsec_per_quarter = int(tempo[-1])
microsec_per_tick = microsec_per_quarter / ticks_per_quarter
seconds_per_tick = microsec_per_tick / 1000000
df['time'] = df['tick']*seconds_per_tick

# empty numpy array where rows are individual keys on a piano, columns are equal to total ticks in the song from track_end
song_total_ticks = int(track_end[0].split(', ')[1])
notes = 127
midi_note_array = np.zeros((127, song_total_ticks))

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
                                  'control_num': df_key_press['control_num'], 'velocity': df_key_press['velocity']})


def map_note(array, note_value, note_start, note_end, velocity):
    # maps midi notes onto the midi array
    # numpy slice isn't inclusive of note_end,
    # as mentioned above, this works b/c note end is the first moment the key is not longer pressed
    array[note_value, note_start:note_end] = velocity

# MIDI MATRIX W/O SUSTAIN PEDAL
for idx, row in df_note_durations.iterrows():
    map_note(midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])

'''Assigning midi pedal arrays'''
# midi maps for pedal presses
df_sus = df_pedals[df_pedals['control_num'] == 64].reset_index(drop=True)
# find duration by tick length to next action
df_sus['duration'] = np.abs(df_sus['tick'].diff(periods=-1))
# extend last action to end of song
df_sus.loc[df_sus.index[-1], 'duration'] = song_total_ticks - df_sus.loc[df_sus.index[-1], 'tick']


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
sus_pedal_press_tick_durations = zip(df_sus_presses['tick'], df_sus_releases['tick'])


def apply_sus_to_slice(start_tick, end_tick, midi_note_array):
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
                array_slice_row[slice[0]: slice[1] + 1] = array_slice_row[slice[0] - 1]
                # if the assigned slice has the same value as the element directly following,
                # assign the end element of the slice to 0 to differentiate the run with the following elements
                # i.e 5, 5, 0, 0, 0, 5 . . . -> 5, 5, 5, 5, 0, 5 instead of 5, 5, 5, 5, 5, 5
                if slice[1] + 1 < len(array_slice_row) and array_slice_row[slice[1]] == array_slice_row[slice[1] + 1]:
                    array_slice_row[slice[1]] = 0
            return array_slice_row
            # return altered input row
        else:
            return array_slice_row
            # return original input row

    midi_slice_with_sus = np.apply_along_axis(extend_notes_in_row, 1, midi_slice)
    midi_note_array[:, start_tick: end_tick] = midi_slice_with_sus
    # apply to every row in array

#todo
# applying suspedal func to durations when the sus pedal is pressed
# for duration in sus_pedal_press_tick_durations:
#     apply_sus_to_slice(duration[0], duration[1], midi_note_array)


# midi_sus_array = np.zeros((1, song_total_ticks))

# midi array for sus
# for idx, row in df_sus.iterrows():
#     # mapping pedal actions to midi_sus_array
#     # note_value param is 0 because midi_sus_array only has one row
#     map_note(midi_sus_array, 0, row['tick'], row['tick'] + int(row['duration']), row['velocity'])
# # pedal is either on or off, so assign all on values to 60
# midi_sus_array[midi_sus_array > 0] = 60


# damper pedal
# df_soft = df_pedals[df_pedals['control_num'] == 67].reset_index(drop=True)
# df_soft['duration'] = np.abs(df_soft['tick'].diff(periods=-1))
# df_soft.loc[df_soft.index[-1], 'duration'] = song_total_ticks - df_soft.loc[df_soft.index[-1], 'tick']
# midi_soft_array = np.zeros((1, song_total_ticks))
#
# for idx, row in df_soft.iterrows():
#     # mapping pedal actions to midi_sus_array
#     # note_value param is 0 because midi_sus_array only has one row
#     map_note(midi_soft_array, 0, row['tick'], row['tick'] + int(row['duration']), row['velocity'])
# midi_soft_array[midi_soft_array > 0] = 60

# ax = sns.heatmap(midi_note_array, linewidths=0)
# plt.show()


resized_midi_note_array = zoom(midi_note_array, (1, 0.12228344731), order=0)
apply_sus_to_slice(0, -1, resized_midi_note_array)


''' 
decoding numpy midi array
'''


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
note_presses_and_releases = np.vstack([find_runs(row, idx) for idx, row in enumerate(resized_midi_note_array)])


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

#JUST NOTE PRESSES AND RELEASES:
sorted_note_presses_and_releases = note_presses_and_releases[note_presses_and_releases[:, 1].argsort()]

def write_midi_line(track, tick, control, channel, control_num, velocity):
    midi_string = ', '.join([str(track), str(tick), str(control), str(channel), str(control_num), str(velocity)])
    midi_string += '\n'
    return midi_string


# recombining midi actions with metadata and end of file strings
midi_out = []
for line in meta_data:
    if 'Tempo' in line:
        new_line = '1, 0, Tempo, 4088860'
        midi_out.append(new_line)
    else:
        midi_out.append(line)
for line in sorted_note_presses_and_releases:
    if line[0] == 128:
        #                              track     tick   control  channel  control_num  velocity
        midi_out.append(write_midi_line(2, int(line[1]), 'Control_c', 0, 64, int(line[2])))
    else:
        #                              track     tick   control  channel  control_num  velocity
        midi_out.append(write_midi_line(2, int(line[1]), 'Note_on_c', 0, int(line[0]), int(line[2])))
for line in track_end:
    midi_out.append(line)


midi_object = pm.csv_to_midi(midi_out)
with open('../data/testing_zoom_midi_io.mid', 'wb') as output_file:
    midi_writer = pm.FileWriter(output_file)
    midi_writer.write(midi_object)

