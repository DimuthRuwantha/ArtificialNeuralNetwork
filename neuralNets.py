import numpy as np
from testing.Testing import Testing

from training.TrainingNetwork import TrainingNetwork

if __name__ == '__main__':
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
    y, sn0, sn1 = ann_training.training_network(input_matrix, output_matrix)
    yy = np.round(y, 2)

    test_data = np.array([1.172, 0.334, 1.194, 0.981, 3.335, 0.979])
    test_service = Testing()
    test_service.testing_network(test_data, sn0, sn1)

   # print(yy)
