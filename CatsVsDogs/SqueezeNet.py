#Code by Florian Bordes for the IFT6266 project : Hiver 2016
#Based on the paper : SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size
#http://arxiv.org/pdf/1602.07360v2.pdf

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
from blocks.utils import print_shape
from blocks.bricks.parallel import Merge
from blocks.serialization import dump, load
#Import functions from fuel
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream

#DEFINE THE FIRE MODULE
def Fire(image_shape, num_input, conv1, conv2, conv3, out, i):
    layers11 = []
    layers12 = []
    layers13 = []
    layers14 = []

    ############# SQUEEZE ###########
    ### 4 Conv 1x1 ###
    layers11.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers11.append(BatchNormalization(name='batch_{}'.format(i)))
    layers11.append(Rectifier())
    conv_sequence11 = ConvolutionalSequence(layers11, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence11.initialize()
    out11 = conv_sequence11.apply(out)
    i = i + 1

    layers12.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers12.append(BatchNormalization(name='batch_{}'.format(i)))
    layers12.append(Rectifier())
    conv_sequence12 = ConvolutionalSequence(layers12, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence12.initialize()
    out12 = conv_sequence12.apply(out)
    i = i + 1

    layers13.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers13.append(BatchNormalization(name='batch_{}'.format(i)))
    layers13.append(Rectifier())
    conv_sequence13 = ConvolutionalSequence(layers13, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence13.initialize()
    out13 = conv_sequence13.apply(out)
    i = i + 1

    layers14.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers14.append(BatchNormalization(name='batch_{}'.format(i)))
    layers14.append(Rectifier())
    conv_sequence14 = ConvolutionalSequence(layers14, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence14.initialize()
    out14 = conv_sequence14.apply(out)
    i = i + 1

    squeezed = T.concatenate([out11, out12, out13, out14], axis=1)

    ####### EXPAND #####
    layers21 = []
    layers22 = []
    layers23 = []
    layers24 = []
    layers31 = []
    layers32 = []
    layers33 = []
    layers34 = []
    num_input2 = conv1 * 4
    ### 4 conv 1x1 ###
    layers21.append(Convolutional(filter_size=(1,1), num_channels=num_input2, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers21.append(BatchNormalization(name='batch_{}'.format(i)))
    layers21.append(Rectifier())
    conv_sequence21 = ConvolutionalSequence(layers21, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence21.initialize()
    out21 = conv_sequence21.apply(squeezed)
    i = i + 1

    layers22.append(Convolutional(filter_size=(1,1), num_channels=num_input2, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers22.append(BatchNormalization(name='batch_{}'.format(i)))
    layers22.append(Rectifier())
    conv_sequence22 = ConvolutionalSequence(layers22, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence22.initialize()
    out22 = conv_sequence22.apply(squeezed)
    i = i + 1

    layers23.append(Convolutional(filter_size=(1,1), num_channels=num_input2, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers23.append(BatchNormalization(name='batch_{}'.format(i)))
    layers23.append(Rectifier())
    conv_sequence23 = ConvolutionalSequence(layers23, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence23.initialize()
    out23 = conv_sequence23.apply(squeezed)
    i = i + 1

    layers24.append(Convolutional(filter_size=(1,1), num_channels=num_input2, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers24.append(BatchNormalization(name='batch_{}'.format(i)))
    layers24.append(Rectifier())
    conv_sequence24 = ConvolutionalSequence(layers24, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence24.initialize()
    out24 = conv_sequence24.apply(squeezed)
    i = i + 1

    ### 4 conv 3x3 ###
    layers31.append(Convolutional(filter_size=(3,3), num_channels=num_input2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers31.append(BatchNormalization(name='batch_{}'.format(i)))
    layers31.append(Rectifier())
    conv_sequence31 = ConvolutionalSequence(layers31, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence31.initialize()
    out31 = conv_sequence31.apply(squeezed)
    i = i + 1

    layers32.append(Convolutional(filter_size=(3,3), num_channels=num_input2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers32.append(BatchNormalization(name='batch_{}'.format(i)))
    layers32.append(Rectifier())
    conv_sequence32 = ConvolutionalSequence(layers32, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence32.initialize()
    out32 = conv_sequence32.apply(squeezed)
    i = i + 1

    layers33.append(Convolutional(filter_size=(3,3), num_channels=num_input2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers33.append(BatchNormalization(name='batch_{}'.format(i)))
    layers33.append(Rectifier())
    conv_sequence33 = ConvolutionalSequence(layers33, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence33.initialize()
    out33 = conv_sequence33.apply(squeezed)
    i = i + 1

    layers34.append(Convolutional(filter_size=(3,3), num_channels=num_input2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers34.append(BatchNormalization(name='batch_{}'.format(i)))
    layers34.append(Rectifier())
    conv_sequence34 = ConvolutionalSequence(layers34, num_channels=num_input2, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence34.initialize()
    out34 = conv_sequence34.apply(squeezed)
    i = i + 1

    #Merge
    return T.concatenate([out21, out22, out23, out24, out31, out32, out33, out34], axis=1)

############## CREATE THE NETWORK ###############
#Define the parameters
#Create the symbolics variable
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

num_epochs = 1000
layers = []

###############FIRST STAGE#######################
#Create the convolutions layers
layers.append(Convolutional(filter_size=(7,7), step=(2,2), num_filters=96, border_mode='half', name='conv_0'))
layers.append(BatchNormalization(name='batch_0'))
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_0'))

convSeq = ConvolutionalSequence(layers, num_channels=3, image_size=(220,220), weights_init=Orthogonal(), use_bias=False, name='ConvSeq')
convSeq.initialize()
out = convSeq.apply(x)

#FIRE MODULES
out1 = Fire((55,55), 96, 16, 16, 16, out, 10)
out2 = Fire((55,55), 128, 16, 16, 16, out1, 25)
out3 = Fire((55,55), 128, 32, 32, 32, out2, 300)
out31 = MaxPooling((3,3), step=(2,2), padding=(1,1), name='poolLow').apply(out3)
out4 = Fire((28,28), 256, 32, 32, 32, out31, 45)
out5 = Fire((28,28), 256, 48, 48, 48, out4, 500)
out6 = Fire((28,28), 384, 48, 48, 48, out5, 65)
out7 = Fire((28,28), 384, 64, 64, 64, out6, 700)
out71 = MaxPooling((3,3), step=(2,2), padding=(1,1), name='poolLow2').apply(out7)
out8 = Fire((14,14), 512, 64, 64, 64, out71, 85)

#LAST LAYERS
conv_layers1 = list([Convolutional(filter_size=(1,1), num_filters=2, name='Convx2'), BatchNormalization(name='batch_vx2'), Rectifier(),
    AveragePooling((14,14), name='MaxPol1')])
conv_sequence1 = ConvolutionalSequence(conv_layers1, num_channels=512, image_size=(14,14), weights_init=Orthogonal(), use_bias=False, name='ConvSeq3')
conv_sequence1.initialize()
out_soft1 = Flattener(name='Flatt1').apply(conv_sequence1.apply(out8))
predict1 = NDimensionalSoftmax(name='Soft1').apply(out_soft1)
cost = CategoricalCrossEntropy(name='Cross1').apply(y.flatten(), predict1).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict1)

#Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])

########### GET THE DATA #####################
stream_train = ServerDataStream(('image_features','targets'), False, port=5512, hwm=40)
stream_valid = ServerDataStream(('image_features','targets'), False, port=5513, hwm=40)

########### DEFINE THE ALGORITHM #############
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Momentum(learning_rate=0.01, momentum=0.9))
extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              TrainingDataMonitoring([cost, error_rate,
                aggregation.mean(algorithm.total_gradient_norm)],
                prefix="train",
                after_epoch=True),
              DataStreamMonitoring([cost, error_rate, error_rate2], stream_valid, prefix="valid", after_epoch=True),
              Checkpoint("catsVsDogs_fire2_new.pkl", after_epoch=True, save_separately=['log']),
              ProgressBar(),
              Printing()]

#Adding a live plot with the bokeh server
extensions.append(Plot(
    'CatsVsDogs_Fire2',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['train_cost', 'valid_cost'],
              ['train_total_gradient_norm']], after_epoch=True))

model = Model(cost)
main_loop = MainLoop(algorithm,data_stream=stream_train,model=model,extensions=extensions)
main_loop.run()
