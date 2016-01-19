import numpy as np
import utils_functions as utils

np.random.seed(123)

#Define a generic class
class Layer(object):

    def __init__(self, n_in, n_units, activ):
        #Init weights
        bound = 1/np.sqrt(n_in)
        self.W = np.random.uniform(low=-bound, high=bound, size=(n_units,n_in))
        #Init bias
        self.b = np.random.uniform(-0.1,0.1,n_units)
        #Choose activation function
        self.activation = activ
        #Select the derivate
        if self.activation == utils.sigmoid:
            self.activation_derivate = utils.sigmoid_derive
        elif self.activation == utils.rect:
            self.activation_derivate = utils.rect_derive

    #Perform forward propagation without batch
    def forward_propagation_norm(self, x):
        self.outputs = self.activation(np.dot(self.W, x) + self.b)
        return self.outputs

    #Perform forward propagation over batch
    def forward_propagation(self, x):
        self.outputs = self.activation(np.tensordot(self.W, x, (1,1)).transpose() + self.b)
        return self.outputs

    #Performan backward propagation
    def backward_propagation(self):
        pass

    #Update the parameters
    def update_parameters(self, l, m, grad_W, grad_b):
        self.W = self.W - l * grad_W - 2 * m * self.W
        self.b = self.b - l * grad_b

#Define a specific class for output
class Output_Layer(Layer):

    def attach(self, Layer_pred):
        self.inputs = Layer_pred

    #Backward operation for an Output Layer without batch
    def backward_propagation_norm(self, y, h1):
        self.cost = self.outputs - utils.onehot(10,y)
        grad_W = np.outer(self.cost, np.transpose(h1))
        grad_b = self.cost.sum(axis=0)
        return grad_W, grad_b

    #Backward operation for an Output Layer with batch
    def backward_propagation(self, y, h1):
        self.cost = self.outputs - utils.onehot(10,y)
        grad_W = np.dot(self.cost.T, h1)
        grad_b = self.cost.sum(axis=0)
        return grad_W, grad_b

#Define a specific class for the Hidden Layer
class Hidden_Layer(Layer):

    #Link the Hidden_Layer to the next Layer
    def __init__(self, n_in, nb_units, activ):
        Layer.__init__(self, n_in, nb_units, activ)

    def attach(self, Layer_next):
        self.layer_next = Layer_next

    #Backward operation for an inner layer without batch
    def backward_propagation_norm(self, x):
        grad_h = np.dot(np.transpose(self.layer_next.W), self.layer_next.cost) * self.activation_derivate(self.outputs)
        grad_W = np.outer(grad_h, np.transpose(x))
        grad_b = grad_h
        return grad_W, grad_b

    #Backward operation for an inner layer with batch
    def backward_propagation(self, x):
        grad_hs = np.tensordot(self.layer_next.W,  self.layer_next.cost, (0,1)).transpose()
        grad_h = grad_hs * self.activation_derivate(self.outputs)
        grad_W = np.dot(grad_h.T, x)
        grad_b = grad_h.sum(axis=0)
        self.cost = grad_h
        return grad_W, grad_b
