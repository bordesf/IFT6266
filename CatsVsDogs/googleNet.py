#Code by Florian Bordes for the IFT6266 course
#Implementation in blocks of GoogleNet:
#http://arxiv.org/pdf/1409.4842v1.pdf

import numpy as np
from theano import tensor
#Import functions from blocks
from blocks.algorithms import GradientDescent, Scale, Adam, Momentum
from blocks.bricks import (MLP, Activation, Rectifier, LinearMaxout, Initializable, FeedforwardSequence, Softmax, BatchNormalization, BatchNormalizedMLP, SpatialBatchNormalization, NDimensionalSoftmax)
from blocks.bricks.conv import (Convolutional, ConvolutionalActivation, ConvolutionalSequence, Flattener, MaxPooling, AveragePooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar, TrainingExtension
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions.predicates import OnLogRecord
from blocks_extras.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.roles import INPUT
from blocks.graph import apply_dropout
from blocks.utils import print_shape
from blocks.bricks.parallel import Merge
from blocks.serialization import dump, load, load_parameter_values
from theano import tensor as T
#Import functions from fuel
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream

class WriteCostExtension(TrainingExtension):

    def after_batch(self, batch):
        self.main_loop.log.current_row['cost'] = abs(self.main_loop.log.status['iterations_done'] - 5) + 3

def inception(image_shape, num_input, conv1, conv2, conv3, conv4, conv5, conv6, out, i):
    layers1 = []
    layers2 = []
    layers3 = []
    layers4 = []
    layers1.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers1.append(BatchNormalization(name='batch_{}'.format(i)))
    layers1.append(Rectifier())
    conv_sequence1 = ConvolutionalSequence(layers1, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence1.initialize()
    out1 = conv_sequence1.apply(out)
    i = i + 1

    layers2.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers2.append(BatchNormalization(name='batch_{}'.format(i)))
    layers2.append(Rectifier())
    i = i + 1
    layers2.append(Convolutional(filter_size=(3,3), num_channels=conv2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers2.append(BatchNormalization(name='batch_{}'.format(i)))
    layers2.append(Rectifier())
    conv_sequence2 = ConvolutionalSequence(layers2, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence2.initialize()
    out2 = conv_sequence2.apply(out)
    i = i + 1

    layers3.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv4, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers3.append(BatchNormalization(name='batch_{}'.format(i)))
    layers3.append(Rectifier())
    i = i + 1
    layers3.append(Convolutional(filter_size=(5,5), num_channels=conv4, num_filters=conv5, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers3.append(BatchNormalization(name='batch_{}'.format(i)))
    layers3.append(Rectifier())
    conv_sequence3 = ConvolutionalSequence(layers3, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence3.initialize()
    out3 = conv_sequence3.apply(out)
    i = i + 1

    layers4.append(MaxPooling((3,3), step=(1,1), padding=(1,1), name='pool_{}'.format(i)))
    layers4.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv6, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers4.append(BatchNormalization(name='batch_{}'.format(i)))
    layers4.append(Rectifier())
    i = i + 1
    conv_sequence4 = ConvolutionalSequence(layers4, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence4.initialize()
    out4 = conv_sequence4.apply(out)
    #Merge
    return T.concatenate([out1, out2, out3, out4], axis=1)

############## CREATE THE NETWORK ###############
#Define the parameters
#Create the stmbolics variable
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

num_epochs = 500
layers = []
###############FIRST STAGE#######################
#Create the convolutions layers
layers.append(Convolutional(filter_size=(7,7), num_filters=32, border_mode='half', name='conv_0'))
layers.append(BatchNormalization(name='batch_0'))
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_0'))

layers.append(Convolutional(filter_size=(1,1), num_filters=64, border_mode='half', name='conv_1'))
layers.append(BatchNormalization(name='batch_1'))
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_1'))
layers.append(Convolutional(filter_size=(3,3), num_filters=192, border_mode='half', name='conv_2'))
layers.append(BatchNormalization(name='batch_2'))
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_2'))

#Create the sequence
conv_sequence = ConvolutionalSequence(layers, num_channels=3, image_size=(160,160), weights_init=Orthogonal(), use_bias=False, name='convSeq')
#Initialize the convnet
conv_sequence.initialize()
#Output the first result
out = conv_sequence.apply(x)

###############SECOND STAGE#####################
out2 = inception((20,20), 192, 64, 96, 128, 16, 32, 32, out, 10)
out3 = inception((20,20), 256, 128, 128, 192, 32, 96, 64, out2, 20)
out31 = MaxPooling((2,2), name='poolLow').apply(out3)

out4 = inception((10,10), 480, 192, 96, 208, 16, 48, 64, out31, 30)
out5 = inception((10,10), 512, 160, 112, 224, 24, 64, 64, out4, 40)
out6 = inception((10,10), 512, 128, 128, 256, 24, 64, 64, out5, 50)
out7 = inception((10,10), 512, 112, 144, 288, 32, 64, 64, out6, 60)
out8 = inception((10,10), 528, 256, 160, 320, 32, 128, 128, out7, 70)
out81 = MaxPooling((2,2), name='poolLow1').apply(out8)

out9 = inception((5,5), 832, 256, 160, 320, 32, 128, 128, out81, 80)
out10 = inception((5,5), 832, 384, 192, 384, 48, 128, 128, out9, 90)
out91 = AveragePooling((5,5), name='poolLow2').apply(out10)

#FIRST SOFTMAX
conv_layers1 = list([MaxPooling((2,2), name='MaxPol'), 
    Convolutional(filter_size=(1,1), num_filters=128, name='Convx2'), Rectifier(),
    MaxPooling((2,2), name='MaxPol1'),
    Convolutional(filter_size=(1,1), num_filters=1024, name='Convx3'), Rectifier(),
    MaxPooling((2,2), name='MaxPol2'),
    Convolutional(filter_size=(1,1), num_filters=2, name='Convx4'), Rectifier(),
    ])
conv_sequence1 = ConvolutionalSequence(conv_layers1, num_channels=512, image_size=(10,10), weights_init=Orthogonal(), use_bias=False, name='ConvSeq3')
conv_sequence1.initialize()
out_soft1 = Flattener(name='Flatt1').apply(conv_sequence1.apply(out5))
predict1 = NDimensionalSoftmax(name='Soft1').apply(out_soft1)
cost1 = CategoricalCrossEntropy(name='Cross1').apply(y.flatten(), predict1).copy(name='cost1')

#SECOND SOFTMAX
conv_layers2 = list([MaxPooling((2,2), name='MaxPol2'), 
    Convolutional(filter_size=(1,1), num_filters=128, name='Convx21'), Rectifier(),
    MaxPooling((2,2), name='MaxPol11'),
    Convolutional(filter_size=(1,1), num_filters=1024, name='Convx31'), Rectifier(),
    MaxPooling((2,2), name='MaxPol21'),
    Convolutional(filter_size=(1,1), num_filters=2, name='Convx41'), Rectifier(),
    ])
conv_sequence2 = ConvolutionalSequence(conv_layers2, num_channels=832, image_size=(10,10), weights_init=Orthogonal(), use_bias=False, name='ConvSeq4')
conv_sequence2.initialize()
out_soft2 = Flattener(name='Flatt2').apply(conv_sequence2.apply(out8))
predict = NDimensionalSoftmax(name='Soft2').apply(out_soft2)
cost2 = CategoricalCrossEntropy(name='Cross2').apply(y.flatten(), predict).copy(name='cost2')

layersf = []
################Last Stage########################
layersf.append(Convolutional(filter_size=(1,1), num_filters=1024, border_mode='half', name='conv_81'))
layersf.append(BatchNormalization(name='batch_81'))
layersf.append(Rectifier())
layersf.append(Convolutional(filter_size=(1,1), num_filters=2, border_mode='half', name='conv_82'))
layersf.append(BatchNormalization(name='batch_82'))
layersf.append(Rectifier())
#Create the sequence
conv_sequence3 = ConvolutionalSequence(layersf, num_channels=1024, image_size=(1,1), weights_init=Orthogonal(), use_bias=False, name='convSeq90')
#Initialize the convnet
conv_sequence3.initialize()
#Output the first result
out_soft3 = conv_sequence3.apply(out91)
outf = Flattener().apply(out_soft3)
predict3 = NDimensionalSoftmax().apply(outf)
cost3 = CategoricalCrossEntropy().apply(y.flatten(), predict3).copy(name='cost3')

cost = cost3 + 0.3 * cost2 + 0.3 * cost1
cost = cost.copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict3)
#Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])

########### GET THE DATA #####################
stream_train = ServerDataStream(('image_features','targets'), False, port=5652, hwm=50)
stream_valid = ServerDataStream(('image_features','targets'), False, port=5653, hwm=50)

########### DEFINE THE ALGORITHM #############
track_cost = TrackTheBest("cost", after_epoch=True, after_batch=False)
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Momentum(learning_rate=0.0001, momentum=0.9))
extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              TrainingDataMonitoring([cost, error_rate,
                aggregation.mean(algorithm.total_gradient_norm)],
                prefix="train",
                after_epoch=True),
              DataStreamMonitoring([cost, error_rate, error_rate2], stream_valid, prefix="valid", after_epoch=True),
              Checkpoint("google_Ortho2_pretrain2_l0001.pkl", after_epoch=True),
              ProgressBar(),
              Printing()]

#Adding a live plot with the bokeh server
extensions.append(Plot(
    'CatsVsDogs160_GoogleNet_Reload2_l0001',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['valid_cost', 'valid_error_rate2'],
              ['train_total_gradient_norm']], after_batch=True))

params = load_parameter_values('GoogleParameters.pkl')
model = Model(cost)
model.set_parameter_values(params)
main_loop = MainLoop(algorithm,data_stream=stream_train,model=model,extensions=extensions)
main_loop.run()
