import os
import py_midicsv as pm
import librosa.display


for filename in os.listdir('/media/zackstrater/New Volume/maestro-v3.0.0/2018'):
    if filename.endswith(".midi"):
        # print(os.path.join(directory, filename))
        print(filename)
        # import midi file, result is a list of strings that contain midi actions
        csv_string = pm.midi_to_csv(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2018', filename))
        meta_data = csv_string[0:7]
        track_end = csv_string[-2:]
        header = meta_data[0].split(',')
        tempo = meta_data[2].split(',')
        ticks_per_quarter = int(header[-1])
        microsec_per_quarter = int(tempo[-1])
        microsec_per_tick = microsec_per_quarter / ticks_per_quarter
        seconds_per_tick = microsec_per_tick / 1000000
        song_total_ticks = int(track_end[0].split(', ')[1])
        microsec_per_quarter = int(tempo[-1])
        print(f'tempo: {microsec_per_quarter}')
        print(f'PPQ: {ticks_per_quarter}')
    else:
        if filename.endswith(".wav"):
            x, sr = librosa.load(os.path.join('/media/zackstrater/New Volume/maestro-v3.0.0/2018', filename), sr=None)
            print(sr)


