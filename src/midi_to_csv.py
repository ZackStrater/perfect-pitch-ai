import py_midicsv as pm
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


csv_string = pm.midi_to_csv('../data/test_midi_csv.midi')

meta_data = csv_string[0:7]
# for line in meta_data:
#     print(line)
midi_data = csv_string[8:-2]


midi_array = np.genfromtxt(midi_data, delimiter=',', dtype=None, encoding=None)

columns = ['track', 'tick', 'control', 'channel', 'control_num', 'value']
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


df_sorted_notes = df_notes.sort_values(['control_num', 'tick']).reset_index(drop=True)
df_key_lift = df_sorted_notes[df_sorted_notes['value'] == 0]
df_key_press = df_sorted_notes[df_sorted_notes['value'] != 0]
print(df_key_press)
print(df_key_lift)

assert df_key_lift.shape[0] == df_key_press.shape[0]
