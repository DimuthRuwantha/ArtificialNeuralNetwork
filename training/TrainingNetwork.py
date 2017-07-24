import numpy as np

from NeuralNetwork.NeuralNetObject import NeuralNetObject
from training.SigmoidFunction import SigmoidFunction


# Created by DimRu on 23-Jun-17


class TrainingNetwork:
    """ Add the class description here """

    def __init__(self):
        self.__input = np.array([[]])
        self.__output = np.array([[]])

        self.__sigmoid = SigmoidFunction()

    def training_network(self, x=list([[]]), y=list([[]])):
        self.__input = np.array(x)
        self.__output = np.array(y)

        neural_object = NeuralNetObject()
        syn0, syn1 = neural_object.create_network(x, y)

        print("Training network")

        for j in range(60000):
            l0 = self.__input
            l1 = self.__sigmoid.nonlin(np.dot(l0, syn0))
            l2 = self.__sigmoid.nonlin(np.dot(l1, syn1))

            l2_error = self.__output - l2

            if j % 10000 == 0:
                # print("Error {}".format(np.mean(np.abs(l2_error))))
                print(".", sep=' ', end='', flush=True)

            l2_delta = l2_error * self.__sigmoid.nonlin(l2, True)

            l1_error = l2_delta.dot(syn1.T)

            l1_delta = l1_error * self.__sigmoid.nonlin(l1, True)

            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
        print("Completed")

        return l2, syn0, syn1
