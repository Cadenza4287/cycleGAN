import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib
import scipy.misc
import copy
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread


def batch_norm(x , name = 'batch_norm'):
    """
    batch norm operate
    :param x: tensor with shape [batch_size , height , width , channels], input tensor
    :param name: string , layer name
    :return: tensor with the same shape with x , batch normed tensor
    """
    return contrib.layers.batch_norm(x , decay= 0.9 , scale= True , updates_collections= None ,
                                     epsilon= 1e-5 , scope= name)

def instance_norm(x , name = 'instance_norm'):
    """
        instance norm operate
        :param x: tensor with shape [batch_size , height , width , channels], input tensor
        :param name: string , layer name
        :return: tensor with the same shape with x , instance normed tensor
        """
    with tf.variable_scope(name):
        depth = x.get_shape()[-1]
        scale = tf.get_variable('scale' , shape= [depth] ,
                                initializer= tf.random_normal_initializer(stddev= 0.02 , dtype= tf.float32))
        offset = tf.get_variable('offset' , [depth] , initializer= tf.constant_initializer(0.0))
        mean , variance = tf.nn.moments(x , axes= [1 , 2] , keep_dims= True)
        epsilon = 1e-5
        norm = scale * (x - mean) * tf.rsqrt(variance + epsilon) + offset

        return norm

def conv2d(x , output_dim , filter_size = 4 ,  stride_size = 2 , stddev = 0.02 ,padding = 'SAME', name = 'conv2d'):
    """
    conv layer
    :param x: tensor with shape [batch_size , height , width , channels], input tensor
    :param output_dim: int , features of conv layer
    :param filter_size: int , the img_size of filter of conv layer
    :param stride_size: int , the img_size of stride of conv layer
    :param stddev: float , the stddev of truncated normal initializer
    :param padding: string , the type of padding of conv layer
    :param name: string , the name of layer
    :return: tensor with the same shape with x , instance normed tensor
    """
    with tf.variable_scope(name):
        weight = tf.get_variable('weight' , shape=[filter_size , filter_size , x.get_shape()[-1] , output_dim] ,
                                 initializer= tf.truncated_normal_initializer(stddev= stddev , dtype= tf.float32))
        # bias = tf.get_variable('bias' , shape= [output_dim] , initializer= tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x , filter= weight , strides= [1 , stride_size , stride_size , 1] , padding= padding)
        # conv = tf.nn.bias_add(conv , bias)

        return conv

def deconv2d(x , output_dim , filter_size = 3 , stride_size = 2 , stddev = 0.02 , padding = 'SAME' , name = 'deconv2d'):
    """
        conv layer
        :param x: tensor with shape [batch_size , height , width , channels], input tensor
        :param output_dim: int , features of deconv layer
        :param filter_size: int , the img_size of filter of deconv layer
        :param stride_size: int , the img_size of stride of deconv layer
        :param stddev: float , the stddev of truncated normal initializer
        :param padding: string , the type of padding of deconv layer
        :param name: string , the name of layer
        :return: tensor with the same shape with x , instance normed tensor
        """
    with tf.variable_scope(name):
        # weight = tf.get_variable('weight' , shape= [filter_size , filter_size , output_shape[-1] , x.get_shape()[-1]])
        #
        # bias = tf.get_variable('bias' , shape= [output_shape[-1]] , initializer= tf.constant_initializer(0.0))
        #
        # deconv = tf.nn.conv2d_transpose(x , filter= weight , output_shape = output_shape ,
        #                                 strides= [1 , stride_size , stride_size , 1] , padding= padding , name = 'deconv')

        return contrib.slim.conv2d_transpose(x , output_dim , kernel_size= filter_size , stride= stride_size ,
                                             padding= padding , activation_fn= None ,
                                             weights_initializer= tf.truncated_normal_initializer(stddev = stddev) ,
                                             biases_initializer= None )

def upsampling(x , multiple = 2 , name = 'upsampling'):
    """

    :param x:
    :param output_dim:
    :param name:
    :return:
    """
    return tf.image.resize_nearest_neighbor(x , (x.get_shape()[1] * multiple , x.get_shape()[2] * multiple) , name= name)

def lrelu(x , alpha = 0.2 , name = 'lrelu'):
    """
    leaky relu operator
    :param x: tensor , input tensor
    :param alpha: float , the alpha of leaky relu
    :param name: string , the name of operator
    :return: tensor with the same shape with x , output tensor
    """
    return tf.nn.leaky_relu(x , alpha= alpha , name= name)

def relu(x):
    """
    relu operator
    :param x: tensor , input tensor
    :return: tensor with the same shape with x , output tensor
    """
    return tf.nn.relu(x)

def tanh(x):
    """
    tanh operator
    :param x: tensor , input tensor
    :return: tensor with the same shape with x , output tensor
    """
    return tf.tanh(x)


def linear(input_tensor , output_dim , layer_name ,stddev = 0.02):
    """
    linear layer
    :param input_tensor: tensor with shape [batch_size , height , width , channels], input tensor
    :param output_dim: int , features of linear layer
    :param layer_name: string , the name of layer
    :param stddev: stddev: float , the stddev of truncated normal initializer
    :return: tensor with the same shape with x , output tensor
    """
    with tf.variable_scope (layer_name):
        weight = tf.get_variable ('weight' ,shape=[input_tensor.get_shape ()[-1] ,output_dim] ,
                                  initializer=tf.random_normal_initializer (stddev = stddev ))
        bias = tf.get_variable ('bias' ,shape=[output_dim] ,initializer=tf.constant_initializer (0.0))
        pre_activate = tf.add (tf.matmul (input_tensor ,weight) ,bias)
    return pre_activate

def res_block(x , dim , filter_size = 3 ,  stride_size = 1 , stddev = 0.02 ,padding = 'SAME', name = 'res_block'):
    """
    residual  block
    :param x: tensor with shape [batch_size , height , width , channels], input tensor
    :param dim: int , features of residual  layer
    :param filter_size: int , the img_size of filter of deconv layer
    :param stride_size: int , the img_size of stride of deconv layer
    :param stddev: float , the stddev of truncated normal initializer
    :param padding: string , the type of padding of deconv layer
    :param name: string , the name of layer
    :return: tensor with the same shape with x , output tensor
    """
    with tf.variable_scope(name_or_scope= name):
        p = int((filter_size - 1) / 2)
        pad0 = tf.pad(x , [[0 , 0] , [p , p] , [p , p] , [0 , 0]] , mode= 'REFLECT')
        hidden0 = instance_norm(conv2d(pad0 , dim , filter_size , stride_size , stddev , padding= 'VALID' , name= 'conv0') , 'ins_norm0')
        pad1 = tf.pad(relu(hidden0) , [[0 , 0] , [p , p] , [p , p] , [0 , 0]] , mode= 'REFLECT')
        hidden1 = instance_norm(conv2d(pad1 , dim , filter_size , stride_size , stddev , padding= 'VALID' , name= 'conv1') , 'ins_norm1')
        return  hidden1 + x

def mean_square_loss(logits , labels):
    """
    calculate the mean square difference between logits and lables
    :param logits: tensor with shape [None , 1], prediction tensor
    :param labels: tensor with the same shape of logits , label tensor
    :return: tensor with type float , loss term
    """
    return tf.reduce_mean(tf.square(logits - labels))

def mean_abs_loss(logits , labels):
    """
    calculate the mean abs difference between logits and lables
    :param logits: tensor with shape [None , 1], prediction tensor
    :param labels: tensor with the same shape of logits , label tensor
    :return: tensor with type float , loss term
    """
    return tf.reduce_mean(tf.abs(logits - labels))

# def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
#     img = imread(image_path)
#     if not is_testing:
#         img = scipy.misc.imresize(img, [load_size, load_size])
#         h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
#         w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
#         img = img[h1:h1+fine_size, w1:w1+fine_size]
#
#         if np.random.random() > 0.5:
#             img_A = np.fliplr(img)
#     else:
#         img = scipy.misc.imresize(img, [fine_size, fine_size])
#
#     img = img/127.5 - 1.
#
#     return img
#
# def imread(path, is_grayscale = False):
#     if (is_grayscale):
#         return _imread(path, flatten=True).astype(np.float)
#     else:
#         return _imread(path, mode='RGB').astype(np.float)

