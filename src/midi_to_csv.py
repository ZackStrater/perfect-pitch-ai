import py_midicsv as pm
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf)

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




print(meta_data)
print(track_end)
