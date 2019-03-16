import numpy as np
import math


def getY(x):
    """
    y = x * x + 5

    Arguments:
    parameter -- x

    Returns:
    y --- x * x + 5
    """

    return 5 + x * x


def generateSample(n=1000, scale=100):

    data = []

    x = scale * ((np.random.random_sample(n, )) - 0.5)

    for i in range(n):
        yi = getY(x[i])
        data.append([x[i], yi])

    return np.array(data)
