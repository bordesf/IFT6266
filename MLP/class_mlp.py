import class_Layers
import utils_functions as utils
import gzip,pickle
import time
import numpy as np
from class_Layers import Hidden_Layer, Output_Layer

#Load MNIST digit dataset
def load_mnist():
     f=gzip.open('mnist.pkl.gz')
     data=pickle.load(f)
     X = data[0][0], data[1][0], data[2][0]
     Y = data[0][1], data[1][1], data[2][1]
     return X,Y

class MLP(object):

    #Initialization of the MLP
    def __init__(self, units, activ):
        self.layers = []
        for i in range(len(units) - 2):
            self.layers.append(Hidden_Layer(units[i], units[i+1], activ))
        self.layers.append(Output_Layer(units[-2], units[-1], utils.softmax_batch))
        for l in range(len(self.layers) - 1):
            self.layers[l].attach(self.layers[l + 1])
        self.layers[-1].attach(self.layers[-1])
        self.X, self.Y = load_mnist()

    #Test function that compute the average error
    def test(self, x, y):
        error = 0
        for i in range(len(x)):
            inputs = []
            inputs.append(x[i:i+1])
            for layer in self.layers:
                inputs.append(layer.forward_propagation(inputs[-1]))
            if y[i] != np.argmax(inputs[-1]):
                error = error + 1.0
        print 'Average error : ', '%.4f'%(error/(len(x)))
        return error/(len(x))

    #Training function
    def train(self, l = 0.01, m = 0, batch_size = 20, epoch_max = 20):
        print 'start train',time.strftime('%d/%m/%y %H:%M',time.localtime())
        nbepoch = 0
        #Create the training, valid and test dataset
        x, xv, xt = self.X
        y, yv, yt = self.Y
        nb_batch = len(x) / batch_size
        errTrain, errValid, errTest, coutTrain, coutValid, coutTest = [], [], [], [], [], []
        #For each epoch
        while(nbepoch < epoch_max):
            print time.strftime('%d/%m/%y %H:%M',time.localtime())
            #Train on each batch
            for i in range(nb_batch - 1):
                inputs = []
                grads = []
                #Compute the bounds
                low = i*batch_size
                up = batch_size*(i+1)
                inputs.append(x[low:up])
                #Forward propagation
                for layer in self.layers:
                    inputs.append(layer.forward_propagation(inputs[-1]))
                #Backward propagation
                grads = [self.layers[-1].backward_propagation(y[low:up], inputs[-2])] + grads
                for index in reversed(range(len(self.layers) - 1)):
                    grads = [self.layers[index].backward_propagation(inputs[index])] + grads
                #Weights and bias update with weight decay
                a = iter(grads)
                for layer in self.layers:
                    grad_W, grad_b = next(a)
                    layer.update_parameters(l, m, grad_W, grad_b)
            nbepoch = nbepoch + 1
            #Test the network
            errTrain.append(self.test(x,y))
            errValid.append(self.test(xv,yv))
            errTest.append(self.test(xt,yt))
        return errTrain, errValid, errTest
