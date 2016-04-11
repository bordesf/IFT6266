### Code by Florian Bordes for the IFT6266 course
### This code save as image the outputs of each layers of the network.

import numpy as np
import theano
from theano import tensor, function
from PIL import Image
#Import functions from blocks
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.bricks import (MLP, Activation, Rectifier, LinearMaxout, Initializable, FeedforwardSequence, Softmax, BatchNormalization, BatchNormalizedMLP, SpatialBatchNormalization, NDimensionalSoftmax)
from blocks.bricks.conv import (Convolutional, ConvolutionalActivation, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks_extras.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.roles import INPUT
from blocks.graph import apply_dropout
from blocks.utils import print_shape
from blocks.serialization import dump, load, load_parameter_values
#Import functions from fuel
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

############## CREATE THE NETWORK ###############
#Define the parameters
#l8 v3
num_epochs = 1000
num_channels = 3
image_shape = (256, 256)
filter_size = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(1,1),(1,1)]
num_filter = [20, 40, 60, 80, 100, 120, 1024, 2]
pooling_sizes = [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(1,1)]
output_size = 2

#Create the stmbolics variable
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

#Get the parameters
conv_parameters = zip(filter_size, num_filter)

#Create the convolutions layers
conv_layers = list(interleave([(Convolutional(
                                  filter_size=filter_size,
                                  num_filters=num_filter,
                                  name='conv_{}'.format(i))
                for i, (filter_size, num_filter) in enumerate(conv_parameters)),
        #(BatchNormalization(name='batch_{}'.format(i)) for i, _ in enumerate(conv_parameters)),
        (Rectifier() for i, (f_size, num_f) in enumerate(conv_parameters)),
        (MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

#Create the sequence
conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape, weights_init=Uniform(width=0.2), use_bias=False)
print conv_sequence
#Initialize the convnet
conv_sequence.initialize()
#Add the Softmax function
out = Flattener().apply(conv_sequence.apply(x))
predict = NDimensionalSoftmax().apply(out)

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
#Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])

#Load the file
test = load('catsVsDogs256_8_v3.pkl')
m = test.model
#Load the test image
pic = Image.open("image.jpg").resize((256, 256))
pix = np.array(pic.getdata()) / 255.0
pix = pix.reshape((pic.size[0], pic.size[1], 3))
pix = pix.reshape((1, 3 ,pic.size[0], pic.size[1]))

#For each layers, save the output as image file
for k in range(6):
    print "Creation model " + str(k)
    y1 = ConvolutionalSequence(conv_layers[0:(k+1)*3], num_channels, image_size=image_shape, use_bias=False).apply(x)
    mo = Model(y1)
    test = mo.set_parameter_values(m.get_parameter_values())
    fprop = function(mo.inputs, mo.outputs[0], allow_input_downcast=True)
    arr = fprop(pix)
    arr = arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2], arr.shape[3]))
    #Normalize to get an image
    for i in range(arr.shape[0]):
        wmin = float(arr[i].min())
        wmax = float(arr[i].max())
        if wmin and wmax:
            arr[i] *= (255.0/float(wmax-wmin))
            arr[i] += abs(wmin)*(255.0/float(wmax-wmin))

    #Plot the outputs
    fig, ax = plt.subplots(nrows=arr.shape[0]/10, ncols=10, sharex=True, sharey=False)
    for i in xrange(arr.shape[0]):
        ax[i/10][i%10].imshow(arr[i], cmap='Greys_r')
        ax[i/10][i%10].axis('off')
        ax[i/10][i%10].autoscale(True)

    plt.axis('off')
    plt.savefig('result'+str(k)+'.png')
    print "Model " + str(k) + " check"
