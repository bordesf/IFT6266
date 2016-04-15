#Code by Florian Bordes for the IFT6266 course
#Example of how to test a model on the test set and create a csv file to
#sbumit on Kaggle website

import numpy as np
from theano import tensor, function
#Import functions from blocks
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.bricks import (MLP, Activation, Rectifier, LinearMaxout, Initializable, FeedforwardSequence, Softmax, BatchNormalization, BatchNormalizedMLP, SpatialBatchNormalization, NDimensionalSoftmax)
from blocks.bricks.conv import (Convolutional, ConvolutionalActivation, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar, TrainingExtension
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.serialization import dump, load, load_parameter_values
from PIL import Image
#Import functions from fuel
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream

############## CREATE THE NETWORK ###############
#Define the parameters
#l8 v3
num_epochs = 1000
num_channels = 3
image_shape = (128, 128)
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
        (BatchNormalization(name='batch_{}'.format(i)) for i, _ in enumerate(conv_parameters)),
        (Rectifier() for i, (f_size, num_f) in enumerate(conv_parameters)),
        (MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

#Create the sequence
conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape, use_bias=False)
#Add the Softmax function
out = Flattener().apply(conv_sequence.apply(x))
predict = NDimensionalSoftmax().apply(out)

#get the test stream
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme, SequentialExampleScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, MaximumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift
size = (128,128)
cats = DogsVsCats(('test',))
stream = DataStream.default_stream(cats, iteration_scheme=SequentialExampleScheme(cats.num_examples))
stream_upscale = MaximumImageDimensions(stream, size, which_sources=('image_features',))
stream_scale = ScaleAndShift(stream_upscale, 1./255, 0, which_sources=('image_features',))
stream_data = Cast(stream_scale, dtype='float32', which_sources=('image_features',))

#Load the parameters of the model
params = load_parameter_values('convnet_parameters.pkl')
mo = Model(predict)
mo.set_parameter_values(params)
#Create the forward propagation function
fprop = function(mo.inputs, mo.outputs[0], allow_input_downcast=True)
tab = []
i = 1
#Get the prediction for each example of the test set
for data in stream_data.get_epoch_iterator():
    predict = np.argmax(fprop(data))
    tab.append([i, predict])
    print str(i) + "," + str(predict)
    i = i + 1
#Save predictions in a csv file
np.savetxt("dump.csv", tab, delimiter=",", fmt='%d')

