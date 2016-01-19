import numpy as np
import matplotlib.pyplot as plt

#Sigmoid function definition
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derive(x):
    return x * (1 - x)

#RELU functions
def rect(x):
    return np.maximum(0,x)

def rect_derive(x):
    return x > 0

#Onehot that return vector
def onehot_norm(m,y):
    t = np.zeros(m)
    t[y]=1
    return t

#Onehot that return a matrix
def onehot(m,y):
    t = np.zeros((len(y), m))
    for i in range(len(y)):
        t[i,y[i]]= 1
    return t

#Def with x as vector
def softmax_norm(x):
    a = np.max(x)
    return np.exp(x - a)/np.sum(np.exp(x - a))

#Softmax with x as matrix
def softmax_batch(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)

#Plot the errors
def plot(errTrain, errValid, errTest):
    plt.figure(1)
    plt.plot(errTrain, 'g-', label='Train')
    plt.plot(errValid, 'r-', label='Valid')
    plt.plot(errTest, 'b-', label='Test')
    plt.xlabel("Epoche")
    plt.ylabel("error")
    plt.title("Taux d'errors MNIST")
    plt.legend(loc='upper right')
    plt.savefig('result_error.png')
