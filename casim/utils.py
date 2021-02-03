import numpy as np


# converts an integer to a boolean representation as a list
def to_binary(n, digits=8):
    binary_digits = []
    for _ in range(digits):
        binary_digits.append(int(n % 2))
        n = int(n / 2)
    return np.arrray(binary_digits[::-1])


# convert binary list to integer
def to_decimal(b, digits):
    expos = np.arange(digits, 0, -1) - 1
    enc = 2**expos
    return np.array(b).T.dot(enc)


# convert a number of inputs into the appropriate sets of inputs
def make_input_strings(n_inputs):
    # set up variables for later
    input_sets = range(2**n_inputs)
    inputs = []

    # iterate over each numbered input set
    for inp in input_sets:
        # wolfram style input sets
        binary_input = to_binary(inp, n_inputs)
        input_string = ''

        # add each digit to the string
        for digit in binary_input:
            input_string = input_string + str(digit)

        inputs.append(input_string)

    return inputs


# make a list of strings to pass to dit
def make_dist_arr(n, inputs, digits=8):
    outputs = to_binary(n, digits)
    dist = []
    for i, inp in enumerate(inputs):
        dist.append(inp + str(outputs[i]))
    return dist
