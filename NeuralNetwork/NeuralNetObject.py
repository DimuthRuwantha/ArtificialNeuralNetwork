import numpy as np


# Created by DimRu on 23-Jun-17


class NeuralNetObject:
    """ Add the class description here """

    def __init__(self):
        self.__input_matrix = list([[]])
        self.__output_matrix = list([[]])
        self.__output_nodes = 0
        self.__input_nodes = 0
        self.__sn0 = list([[]])
        self.__sn1 = list([[]])

    def create_network(self, input_matrix=list([[]]), output_matrix=list([[]])):
        self.__output_nodes = len(output_matrix[0])
        self.__input_nodes = len(input_matrix[0])
        self.__input_matrix = input_matrix
        self.__output_matrix = output_matrix

        np.random.seed(1)
        # syn0 = 2 * np.random.random((3, 4)) - 1
        # syn1 = 2 * np.random.random((4, 1)) - 1

        syn0 = 2 * np.random.random((self.__input_nodes, self.__input_nodes + 1)) - 1
        syn1 = 2 * np.random.random((self.__input_nodes + 1, self.__output_nodes)) - 1

        return syn0, syn1
