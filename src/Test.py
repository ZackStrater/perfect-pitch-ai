
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


new_arr = np.array([[ 8,  0,  0,  0,  0],
                    [ 8, 32, 32, 32, 32],
                     [ 8,  0,  0,  0,  0],
                     [ 8,  0,  0,  0,  0]])

print(new_arr)
apply_sus_to_slice(0, -1, new_arr)
print(new_arr)