
import numpy as np
import time
from scipy.ndimage import zoom


array = np.array([[1, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 9, 0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 9, 0, 0, 0, 1, 1, 0],
                 [1, 1, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 9, 0, 0, 0, 1, 0, 1],
                 [1, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 9, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 9, 0, 0, 0, 1, 0, 8]])


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


def check_nums(array):
    counter = 0
    for row in array:
        for elem in row:
            if elem == 0:
                counter += 1
    return counter


resized_array = zoom(array, (1, 0.817), order=0)
# print(array.shape)
# apply_sus_to_slice(0, -1, array)
# print(array)
# print(resized_array)


def toy_split_into_intervals(array, sub_array_length):
    array_length = array.shape[1]
    print(array_length)
    print(sub_array_length)
    print(array_length//sub_array_length)
    mod = array_length % sub_array_length
    print(f'mod {mod}')
    pad = sub_array_length - mod
    print(f'pad {pad}')
    padded_array = np.pad(array, ((0, 0), (0, pad)))
    print(padded_array)
    padded_array_length = padded_array.shape[1]
    print(padded_array.shape[1])
    num_intervals = padded_array_length/sub_array_length
    split_arrays = np.array_split(padded_array, num_intervals, axis=1)
    return split_arrays


test = toy_split_into_intervals(array, 9)
print(test)

def split_into_intervals(array, time_interval, sample_rate, hop_length=512):
    # get sample ticks per time interval
    ticks_per_interval = time_interval * sample_rate / hop_length
    subarray_length = ticks_per_interval

    array_length = array.shape[1]

    # number of columns left over
    last_interval_length = array_length % subarray_length
    # how many columns to add to make another full interval
    padding = subarray_length - last_interval_length
    padded_array = np.pad(array, ((0, 0), (0, padding)))
    padded_array_length = padded_array.shape[1]

    num_intervals = padded_array_length/subarray_length
    # split array into subsections ultimately based on time_interval and sample_rate
    split_arrays = np.array_split(padded_array, num_intervals, axis=1)
    return split_arrays