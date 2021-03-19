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

# empty numpy array where rows are individual keys on a piano, columns are equal to total ticks in the song
song_total_ticks = df['tick'].iloc[-1]
notes = 127
midi_note_array = np.zeros((127, song_total_ticks))

# extracting just the note presses and releases
mask = df['control'] == ' Note_on_c'
df_notes = df[mask]
df_pedals = df[~mask]
df_sustain = df_pedals[df_pedals['control_num'] == 64]
df_soft = df_pedals[df_pedals['control_num'] == 67]

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


for idx, row in df_note_durations.iterrows():
    map_note(midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])

# ax = sns.heatmap(midi_note_array, linewidths=0)
# plt.show()

def find_runs(x, row_index):
    # finds continuous segments of note presses and returns the value, start, length, and velocity for each note

    n = x.shape[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True

    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run values
    run_values = x[loc_run_start]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    notes_mask = run_values != 0
    note_starts = run_starts[notes_mask]
    note_lengths = run_lengths[notes_mask]
    note_velocities = run_values[notes_mask]

    # need to make a np array with note value (passed in as row_index) that is the same length as
    # the number of notes found for that row
    note_values = np.full(len(note_starts), row_index)
    return np.dstack((note_values, note_starts, note_lengths, note_velocities))[0]


# gathering note presses for each row (note) in midi array
note_presses = np.vstack([find_runs(row, idx) for idx, row in enumerate(midi_note_array)])

# note releases are reported as distinct events
# key lifts are calculated by taking note press tick and adding note duration
# key lift occurs after note duration end (start + length)
note_releases = note_presses.copy()
note_releases[:, 1] += note_releases[:, 2]
# note length and velocity is 0 for key lift
note_releases[:, 2:4] = 0

# combine presses and releases and sort in order of the tick at which they happen (which is the appropriate order
# for the midi actions to be written to a midi file)
presses_and_releases = np.vstack([note_presses, note_releases])
sorted_presses_and_releases = presses_and_releases[presses_and_releases[:, 1].argsort()]

def write_midi_line(track, tick, control, channel, control_num, velocity):
    midi_string = ', '.join([str(track), str(tick), str(control), str(channel), str(control_num), str(velocity)])
    midi_string += '\n'
    return midi_string


# recombining midi actions with metadata and end of file strings
midi_out = []
for line in meta_data:
    midi_out.append(line)
for line in sorted_presses_and_releases:
#                                  track     tick   control  channel  control_num  velocity
    midi_out.append(write_midi_line(2, int(line[1]), 'Note_on_c', 0, int(line[0]), int(line[3])))
for line in track_end:
    midi_out.append(line)

midi_object = pm.csv_to_midi(midi_out)
with open('../data/testing_midi_io.mid', 'wb') as output_file:
    midi_writer = pm.FileWriter(output_file)
    midi_writer.write(midi_object)

