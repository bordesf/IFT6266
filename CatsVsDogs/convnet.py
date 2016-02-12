#Code by Florian Bordes for the IFT6266 project
#Inspired by LeNet Mnist https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
import numpy as np
from theano import tensor
#Import functions from blocks
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence, Softmax)
from blocks.bricks.conv import (ConvolutionalActivation, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks_extras.extensions.plot import Plot
#Import functions from fuel
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream

############## CREATE THE NETWORK ###############
#Define the parameters
num_epochs = 100
num_channels = 3
image_shape = (128, 128)
filter_size = [(5,5),(5,5),(5,5)]
num_filter = [20, 50, 80]
pooling_sizes = [(2,2),(2,2),(2,2)]
mlp_hiddens = [1000]
output_size = 2
activation = [Rectifier().apply for _ in num_filter]
mlp_activation = [Rectifier().apply for _ in mlp_hiddens] + [Softmax().apply]

#Create the symbolics variable
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

#Get the parameters
conv_parameters = zip(activation, filter_size, num_filter)

#Create the convolutions layers
conv_layers = list(interleave([(ConvolutionalActivation(
                                  filter_size=filter_size,
                                  num_filters=num_filter,
                                  activation=activation,
                                  name='conv_{}'.format(i))
                for i, (activation, filter_size, num_filter)
                in enumerate(conv_parameters)),
        (MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

#Create the sequence
conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape, weights_init=Uniform(width=0.2), biases_init=Constant(0.))
#Initialize the convnet
conv_sequence.initialize()
#Add the MLP
top_mlp_dims = [np.prod(conv_sequence.get_dim('output'))] + mlp_hiddens + [output_size]
out = Flattener().apply(conv_sequence.apply(x))
mlp = MLP(mlp_activation, top_mlp_dims, weights_init=Uniform(0, 0.2),
          biases_init=Constant(0.))
#Initialisze the MLP
mlp.initialize()
#Get the output
predict = mlp.apply(out)

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
#Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])

########### GET THE DATA #####################
stream_train = ServerDataStream(('image_features','targets'), False, port=5550)
stream_valid = ServerDataStream(('image_features','targets'), False, port=5551)

########### DEFINE THE ALGORITHM #############
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())
extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              DataStreamMonitoring([cost, error_rate, error_rate2], stream_valid, prefix="valid"),
              TrainingDataMonitoring([cost, error_rate,
                aggregation.mean(algorithm.total_gradient_norm)],
                prefix="train",
                after_epoch=True),
              Checkpoint("catsVsDogs128.pkl"),
              ProgressBar(),
              Printing()]

#Adding a live plot with the bokeh server
extensions.append(Plot(
    'CatsVsDogs_128_Layer3',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['valid_cost', 'valid_error_rate2'],
              ['train_total_gradient_norm']], after_epoch=True))

model = Model(cost)
main_loop = MainLoop(algorithm,data_stream=stream_train,model=model,extensions=extensions)
main_loop.run()
