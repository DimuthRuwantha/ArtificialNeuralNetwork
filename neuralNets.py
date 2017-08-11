import numpy as np
from testing.Testing import Testing

from training.TrainingNetwork import TrainingNetwork
import util


def get_matrices(training_set=[]):
    matrix_list = []
    output_list = []
    for lst in training_set:
        matrix_list.append(lst[:-1])
        output_list.append(lst[6])
    output_array = process_output_matrix(output_list)
    input_array = np.array(matrix_list)
    return input_array, output_array


def process_output_matrix(output_list=[]):
    out = []
    for fault in output_list:

        if fault == "A to B to Grd":
            out.append([1, 1, 0, 1])

        elif fault == 'A to B':
            out.append([1, 1, 0, 0])

        elif fault == 'C to A':
            out.append([1, 0, 1, 0])

        elif fault == 'A to Grd':
            out.append([1, 0, 0, 1])

        elif fault == 'B to C to Grd':
            out.append([0, 1, 1, 1])

        elif fault == 'B to C':
            out.append([0, 1, 1, 0])

        elif fault == 'B to Grd':
            out.append([0, 1, 0, 1])

        elif fault == 'C to Grd':
            out.append([0, 0, 1, 1])

        elif fault == 'A to C to Grd':
            out.append([1, 0, 1, 1])

        elif fault == 'no Fault':
            out.append([0, 0, 0, 0])
        else:
            out.append([1, 1, 1, 1])
    return np.array(out)


def check_accuracy(results, output):
    accuracy = 0
    output_length = len(output)
    for i in range(output_length):
        k = 0
        for j in range(len(output[i])):
            if results[i][j] == output[i][j]:
                k += 1
            if k == 4:
                accuracy += 1
    print('total no of inputs : ', output_length)
    print("Accurately predicted : ", accuracy)
    return (accuracy / output_length) * 100


if __name__ == '__main__':
    training, testing, classes = util.file_reader()
    input_matrix1, output_matrix1 = get_matrices(testing)

    input_matrix = np.array([[0.997, 0.9991, 0.9985, 0.9978, 0.9988, 0.9984],
                             [0.334, 1.194, 1.172, 3.335, 0.981, 0.979],
                             [1.172, 0.334, 1.194, 0.981, 3.335, 0.979],
                             [1.194, 1.172, 0.334, 0.981, 0.979, 3.335],
                             [0.471, 0.650, 0.986, 5.379, 5.379, 0.983],
                             [0.986, 0.471, 0.650, 0.984, 5.379, 5.379],
                             [0.471, 0.986, 0.650, 5.379, 0.984, 5.379],
                             [0.205, 0.205, 1.188, 7.187, 7.855, 0.985],
                             [1.188, 0.205, 0.205, 0.985, 7.187, 7.855],
                             [0.205, 1.188, 0.205, 7.187, 0.985, 7.855]])

    output_matrix = np.array([[0, 0, 0, 0],
                              [1, 0, 0, 1],
                              [0, 1, 0, 1],
                              [0, 0, 1, 1],
                              [1, 1, 0, 0],
                              [0, 1, 1, 0],
                              [1, 0, 1, 0],
                              [1, 1, 0, 1],
                              [0, 1, 1, 1],
                              [1, 0, 1, 1]])

    ann_training = TrainingNetwork()
    y, weights_input_hidden, weights_hidden_output = ann_training.training_network(input_matrix, output_matrix)
    y_rounded = np.round(y, 2)
    print("weights from 1st layer to hidden layer:")
    print(weights_input_hidden)
    print("weights from hidden layer to output layer:")
    print(weights_hidden_output)

    test_data = input_matrix1  # np.array([1.188, 0.205, 0.205, 0.985, 7.187, 7.855])    # B to C to Grd
    test_service = Testing()
    result = test_service.testing_network(test_data, weights_input_hidden, weights_hidden_output)
    result = np.round(result)

    accur = check_accuracy(result, output_matrix1)
    print(accur)
