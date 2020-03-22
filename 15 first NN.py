import numpy as np

# m1 = length1
# m2 = length2
# w1 = weight1
# w2 = weight2
# b = bios

# sigmoid = 1/(1+e^-x)


def NeuralNetwork(m1, w1, m2, w2, b):
    a = m1*w1 + m2*w2 + b
    sigmoid(a)


def sigmoid(a):
    sig = 1/(1+np.exp(-a))
    # print(sig)


length = [3, 2, 4, 3, 3.5, 2, 5.5, 1, 4.5]
width = [1.5, 1, 1.5, 1, .5, .5, 1, 1, 1]


if len(length) == len(width):
    for i in range(len(length)):

        m1 = length[i]
        m2 = length[i]
        w1 = np.random.randn()
        w2 = np.random.randn()
        b = np.random.randn()

        NeuralNetwork(m1, w1, m2, w2, b)


def cost(b):
    return (b-4)**2


def num_slope(b):
    h = 0.0001
    print((cost(b+h)-cost(b))/h)


def slope(b):
    return 2*(b-4)


num_slope(3)