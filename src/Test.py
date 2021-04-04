
import numpy as np
import time

def consecutive(data):
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

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
        return np.split(data, np.where(np.diff(data) != 1)[0] + 1)

    def extend_notes_in_row(array_slice_row):
        # get indexes where there are 0's
        zeros_arg = np.argwhere(array_slice_row == 0)
        zeros_index = np.squeeze(zeros_arg)
        # find consecutive runs of 0's
        zeros_runs = consecutive(zeros_index)  # TODO might need an if statement here
        # get start and ends of runs of 0's as list of tuples
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

    midi_slice_with_sus = np.apply_along_axis(extend_notes_in_row, 1, midi_slice)
    midi_note_array[:, start_tick: end_tick] = midi_slice_with_sus



def check_nums(array):
    counter = 0
    for row in array:
        for elem in row:
            if elem == 0:
                counter += 1
    return counter

# print(check_nums(array))
# arr2 = np.random.randint(2, size=(10**2, 10**1))
# print(arr2[0:3, 0:10])
# apply_sus_to_slice(0, arr2.shape[1] +1 , arr2)
# print(arr2[0:3, 0:10])


powers = [2, 3, 4, 5]
for_loop_times = []
numpy_times = []
for p in powers:
    arr = np.random.randint(2, size=(100, 10**p))
    t0 = time.time()
    check_nums(arr)
    t1 = time.time()
    for_loop_times.append(t1-t0)
    print(arr.shape)

    t2 = time.time()
    print(arr[0:3, 0:10])
    apply_sus_to_slice(0, array.shape[1], arr)
    print(arr[0:3, 0:10])
    t3 = time.time()
    numpy_times.append(t3-t2)
    print(t3-t2)

    print('\n\n')

print(powers)
print(for_loop_times)
print(numpy_times)