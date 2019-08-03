import numpy as np
def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+np.exp(-x))

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def cost(b):
    return(b-4)**4

def num_slope(b):
    h = 0.0001
    return(cost(b+h)-cost(b))/h

def slope(b):
    return 2*(b-4)
