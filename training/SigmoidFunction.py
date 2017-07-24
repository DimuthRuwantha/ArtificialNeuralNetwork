import numpy as np


# Created by DimRu on 23-Jun-17


class SigmoidFunction:
    """ Add the class description here """

    def __init__(self):
        pass

    def nonlin(self, x, derive=False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
