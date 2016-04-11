### Code by Florian Bordes for the IFT6266 course
### The DeepDream code part come from the Google ipython notebook : 
### https://github.com/google/deepdream
### I have modified it to be use with theano expression instead of a Caffe model


import numpy as np
import theano
from theano import tensor, function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
#MLP
import class_Layers
from class_Layers import Hidden_Layer, Output_Layer, ConvPool
import utils_functions as utils
import scipy.ndimage as nd
from PIL import Image

#### CREATE THE MODEL #####
class WriteCostExtension(TrainingExtension):

    def after_batch(self, batch):
        self.main_loop.log.current_row['cost'] = abs(self.main_loop.log.status['iterations_done'] - 5) + 3

def inception(image_shape, num_input, conv1, conv2, conv3, conv4, conv5, conv6, out, i):
    layers1 = []
    layers2 = []
    layers3 = []
    layers4 = []
    layers1.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv1, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers1.append(Rectifier())
    conv_sequence1 = ConvolutionalSequence(layers1, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence1.initialize()
    out1 = conv_sequence1.apply(out)
    i = i + 1

    layers2.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv2, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers2.append(Rectifier())
    i = i + 1
    layers2.append(Convolutional(filter_size=(3,3), num_channels=conv2, num_filters=conv3, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers2.append(Rectifier())
    conv_sequence2 = ConvolutionalSequence(layers2, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence2.initialize()
    out2 = conv_sequence2.apply(out)
    i = i + 1

    layers3.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv4, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers3.append(Rectifier())
    i = i + 1
    layers3.append(Convolutional(filter_size=(5,5), num_channels=conv4, num_filters=conv5, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
    layers3.append(Rectifier())
    conv_sequence3 = ConvolutionalSequence(layers3, num_channels=num_input, image_size=image_shape, weights_init=Orthogonal(), use_bias=False, name='convSeq_{}'.format(i))
    conv_sequence3.initialize()
    out3 = conv_sequence3.apply(out)
    i = i + 1

    layers4.append(MaxPooling((3,3), step=(1,1), padding=(1,1), name='pool_{}'.format(i)))
    layers4.append(Convolutional(filter_size=(1,1), num_channels=num_input, num_filters=conv6, image_size=image_shape, border_mode='half', name='conv_{}'.format(i)))
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
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_0'))

layers.append(Convolutional(filter_size=(1,1), num_filters=64, border_mode='half', name='conv_1'))
layers.append(Rectifier())
layers.append(Convolutional(filter_size=(3,3), num_filters=192, border_mode='half', name='conv_2'))
layers.append(Rectifier())
layers.append(MaxPooling((3,3), step=(2,2), padding=(1,1), name='pool_2'))

#Create the sequence
conv_sequence = ConvolutionalSequence(layers, num_channels=3, image_size=(None,None), weights_init=Orthogonal(), use_bias=False, name='convSeq')
#Initialize the convnet
#Output the first result
out = conv_sequence.apply(x)

###############SECOND STAGE#####################
out2 = inception((None,None), 192, 64, 96, 128, 16, 32, 32, out, 10)
out3 = inception((None,None), 256, 128, 128, 192, 32, 96, 64, out2, 20)
out31 = MaxPooling((2,2), name='poolLow').apply(out3)
out4 = inception((None,None), 480, 192, 96, 208, 16, 48, 64, out31, 30)
out5 = inception((None,None), 512, 160, 112, 224, 24, 64, 64, out4, 40)

out6 = inception((None,None), 512, 128, 128, 256, 24, 64, 64, out5, 50)
out7 = inception((None,None), 512, 112, 144, 288, 32, 64, 64, out6, 60)
out8 = inception((None,None), 528, 256, 160, 320, 32, 128, 128, out7, 70)
out81 = MaxPooling((20,20), name='poolLow1').apply(out8)
out9 = inception((None,None), 832, 256, 160, 320, 32, 128, 128, out81, 80)
out10 = inception((None,None), 832, 384, 192, 384, 48, 128, 128, out9, 90)
out91 = AveragePooling((5,5), name='poolLow2').apply(out10)

#Load the model
params = load_parameter_values('catsVsDogs_google_new_Ortho.pkl')
model1 = Model(out8)
model1.set_parameter_values(params)
cost = model1.outputs[0].sum()**2
#cost = (T.dot(model1.outputs[0].flatten().T,model1.outputs[0].flatten())**2).sum()
f_prop = theano.function(model1.inputs, model1.outputs[0])
grad = T.grad(cost, model1.inputs)
f_grad = theano.function(model1.inputs, grad)

######DEEP DREAM CODE #############
def preprocess(img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - img.mean()
def deprocess(img):
    return np.dstack((img + img.mean())[::-1])

def make_step(x, step_size=1.5, jitter=32):
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    x[0] = np.roll(np.roll(x[0], ox, -1), oy, -2) # apply jitter shift
    g = f_grad(x)
    # apply normalized ascent step to the input image
    x += step_size/np.abs(g[0]).mean() * g[0]
    x = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    bias = x.mean()
    x = np.clip(x, -bias, 255-bias)
    return x

def deepdream(base_img, iter_n=5, octave_n=4, octave_scale=1.4, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        x = np.array((1,3,h,w)) # resize the network's input image size
        x = octave_base+detail
        for i in xrange(iter_n):
            print h,w
            make_step(x.reshape(1,3,h,w))
            # visualization
            vis = deprocess(x)
            #showarray(vis)
            #print octave, i, end, vis.shape
        # extract details produced on the current octave
        detail = x-octave_base
    # returning the resulting image
    return deprocess(x)

img = np.float32(Image.open('image.jpg').resize((912,684)))

#Making a video
'''
frame = img
frame_i = 0
h, w = frame.shape[:2]
s = 0.05
for i in xrange(100):
    frame = deepdream(frame)
    frame = np.uint8(np.clip(frame, 0, 255))
    Image.fromarray(np.uint8(frame)).save("frames2/%04d.jpg"%frame_i)
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1
'''
#img = preprocess(img).reshape(1,3,160,160)
img = deepdream(img)
a = np.uint8(np.clip(img, 0, 255))
a = np.roll(a, 100)
#plt.axis('off')
#plt.title("inception (4e) : Gram cost")
#plt.imshow(a)
#plt.savefig('dream_udem1.jpg', bbox_inches='tight')
#Image.fromarray(a).save('dream_udem3.jpg')

