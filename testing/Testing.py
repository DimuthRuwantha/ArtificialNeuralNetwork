import numpy as np
from training.SigmoidFunction import SigmoidFunction

# Created by DimRu on 24-Jul-17


class Testing:
    """ Add the class description here """

    def __init__(self):
        self.__sn0 = list([[]])
        self.__sn1 = list([[]])
        self.__sigmoid = SigmoidFunction()

    def testing_network(self, test_data, sn0, sn1):

        self.__sn0 = sn0
        self.__sn1 = sn1
        ll0 = test_data
        var = np.dot(ll0, sn0)
        # print(np.round(var, 2))
        ll1 = self.__sigmoid.nonlin(np.dot(ll0, sn0))
        # print(np.round(ll1, 2))
        # print(np.round(np.dot(ll1, sn1), 2))
        l2 = self.__sigmoid.nonlin(np.dot(ll1, sn1))

        # print("output result \n {}".format(np.round(l2, 2)))
        return l2

