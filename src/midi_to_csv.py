import py_midicsv as pm
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#np.set_printoptions(threshold=np.inf)

csv_string = pm.midi_to_csv('../data/test_midi_csv.midi')

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
print(df_notes)
df_pedals = df[~mask]
df_sustain = df_pedals[df_pedals['control_num'] == 64]
df_soft = df_pedals[df_pedals['control_num'] == 67]


df_sorted_notes = df_notes.sort_values(['control_num', 'tick']).reset_index(drop=True)
df_key_lift = df_sorted_notes[df_sorted_notes['velocity'] == 0].reset_index()
df_key_press = df_sorted_notes[df_sorted_notes['velocity'] != 0].reset_index()

assert df_key_lift.shape[0] == df_key_press.shape[0]
assert df_key_press['control_num'].equals(df_key_lift['control_num'])

df_note_durations = pd.DataFrame({'start_tick': df_key_press['tick'], 'end_tick': df_key_lift['tick'],
                                  'control_num': df_key_press['control_num'], 'velocity': df_key_press['velocity']})

print(df_note_durations)

def map_note(array, note_value, note_start, note_end, velocity):
    array[note_value, note_start:note_end+1] = velocity


for idx, row in df_note_durations.iterrows():
    map_note(midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])


def write_midi_line(track, tick, control, channel, control_num, velocity):
    midi_string = ', '.join([str(track), str(tick), str(control), str(channel), str(control_num), str(velocity)])
    midi_string += '\n'
    return midi_string

# print(meta_data)
# print(write_midi_line(2, 447165, 'Control_c', 0, 64, 0))
# print(track_end)

# x, y = np.nonzero(midi_note_array)
# print(x)
# print(y)
# print(y.shape)


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

print('\n\n\n\n\n\n')

def find_runs(x):
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

    return note_starts, note_lengths, note_velocities

print('\n\n\n\n\n\n')

print(find_runs(midi_note_array[55]))
print(midi_note_array[55, 73381:(73381+67)])




# midi_out = []
# for line in meta_data:
#     midi_out.append(line)
# midi_out.append(write_midi_line(2, 0, 'Note_on_c', 0, 50, 60))
# midi_out.append(write_midi_line(2, 1000, 'Note_on_c', 0, 50, 0))
# midi_out.append(write_midi_line(2, 1000, 'Note_on_c', 0, 52, 60))
# midi_out.append(write_midi_line(2, 2000, 'Note_on_c', 0, 52, 0))
# midi_out.append(write_midi_line(2, 2000, 'Note_on_c', 0, 53, 60))
# midi_out.append(write_midi_line(2, 3000, 'Note_on_c', 0, 53, 0))
# for line in track_end:
#     midi_out.append(line)
# print(midi_out)
#
# midi_object = pm.csv_to_midi(midi_out)
# with open('../data/testing_midi_io.mid', 'wb') as output_file:
#     midi_writer = pm.FileWriter(output_file)
#     midi_writer.write(midi_object)