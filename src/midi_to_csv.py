import py_midicsv as pm
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
#np.set_printoptions(threshold=np.inf)

csv_string = pm.midi_to_csv('../data/test_midi_csv.midi')

print(csv_string)

meta_data = csv_string[0:7]
track_end = csv_string[-2:]
# for line in meta_data:
#     print(line)
midi_data = csv_string[8:-2]


midi_array = np.genfromtxt(midi_data, delimiter=',', dtype=None, encoding=None)

columns = ['track', 'tick', 'control', 'channel', 'control_num', 'velocity']
df = pd.DataFrame(midi_array)
df.columns = columns

header = meta_data[0].split(',')
tempo = meta_data[2].split(',')

ticks_per_quarter = int(header[-1])
microsec_per_quarter = int(tempo[-1])
microsec_per_tick = microsec_per_quarter / ticks_per_quarter
seconds_per_tick = microsec_per_tick / 1000000

df['time'] = df['tick']*seconds_per_tick

song_total_ticks = df['tick'].iloc[-1]
notes = 127
midi_note_array = np.zeros((127, song_total_ticks))

mask = df['control'] == ' Note_on_c'
df_notes = df[mask]
df_pedals = df[~mask]
df_sustain = df_pedals[df_pedals['control_num'] == 64]
df_soft = df_pedals[df_pedals['control_num'] == 67]
print(df_notes.head(50))

df_sorted_notes = df_notes.sort_values(['control_num', 'tick']).reset_index(drop=True)
df_key_lift = df_sorted_notes[df_sorted_notes['velocity'] == 0].reset_index()
df_key_press = df_sorted_notes[df_sorted_notes['velocity'] != 0].reset_index()

assert df_key_lift.shape[0] == df_key_press.shape[0]
assert df_key_press['control_num'].equals(df_key_lift['control_num'])


df_note_durations = pd.DataFrame({'start_tick': df_key_press['tick'], 'end_tick': df_key_lift['tick'],
                                  'control_num': df_key_press['control_num'], 'velocity': df_key_press['velocity']})


def map_note(array, note_value, note_start, note_end, velocity):
    # numpy slice isn't inclusive of note_end,
    # but it works b/c note end is the first moment the key is not longer pressed
    array[note_value, note_start:note_end] = velocity


for idx, row in df_note_durations.iterrows():
    map_note(midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])

# print(meta_data)
# print(write_midi_line(2, 447165, 'Control_c', 0, 64, 0))
# print(track_end)


print('\n\n\n\n\n\n')

def find_runs(x, row_index):
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

print('\n\n\n\n\n\n')

# print(find_runs(midi_note_array[55], 55))
# print(midi_note_array[55, 73381:(73381+66)])
# print(midi_note_array[31, 292367:(292367+47)])
# print(midi_note_array[31, (292367+4)])

note_presses = np.vstack([find_runs(row, idx) for idx, row in enumerate(midi_note_array)])
note_releases = note_presses.copy()

# key lifts are calculated by taking note press tick and adding note duration
# key lift occurs after note duration end (start + length + 1)
note_releases[:, 1] += note_releases[:, 2]
# note length and velocity is 0 for key lift
note_releases[:, 2:4] = 0

presses_and_releases = np.vstack([note_presses, note_releases])
print(presses_and_releases)
sorted_presses_and_releases = presses_and_releases[presses_and_releases[:, 1].argsort()]
print(sorted_presses_and_releases)

def write_midi_line(track, tick, control, channel, control_num, velocity):
    midi_string = ', '.join([str(track), str(tick), str(control), str(channel), str(control_num), str(velocity)])
    midi_string += '\n'
    return midi_string



midi_out = []
for line in meta_data:
    midi_out.append(line)
for line in sorted_presses_and_releases:
#   track    tick     control  channel  control_num  velocity
    midi_out.append(write_midi_line(2, int(line[1]), 'Note_on_c', 0, int(line[0]), int(line[3])))
for line in track_end:
    midi_out.append(line)
print(midi_out)

midi_object = pm.csv_to_midi(midi_out)
with open('../data/testing_midi_io.mid', 'wb') as output_file:
    midi_writer = pm.FileWriter(output_file)
    midi_writer.write(midi_object)