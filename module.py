import tensorflow as tf
from ops import *

def disc(image , df_dim , reuse = False , name = 'disc'):
    """
    discrimitor layer
    :param image: tensor with shape [batch_size , height , width , channels], input image
    :param df_dim: int , the dimention of fitst discrimator hidden layer
    :param reuse: bool , whether reuse the variables
    :param name: string , the name of layer
    :return: tensor , output tensor of layer
    """
    with tf.variable_scope(name_or_scope= name , reuse = reuse):

        conv0 = lrelu(conv2d(image , df_dim , name= 'conv0'))#128
        conv1 = lrelu(instance_norm(conv2d(conv0 , df_dim * 2 , name= 'conv1') , name = 'ins_norm1'))#64
        conv2 = lrelu(instance_norm(conv2d(conv1 , df_dim * 4 , name= 'conv2') , name = 'ins_norm2'))#32
        conv3 = lrelu(instance_norm(conv2d(conv2 , df_dim * 8 , stride_size= 1 ,  name= 'conv3') , name = 'ins_norm3'))
        conv4 = conv2d(conv3 , output_dim= 1 , stride_size= 1 , name= 'conv4')

        return conv4

def gen_resnet(image ,gf_dim ,reuse = False ,name ='gen'):
    """
        genrator layer
        :param image: tensor with shape [batch_size , height , width , channels], input image
        :param gf_dim: int , the dimention of fitst generator hidden layer
        :param reuse: bool , whether reuse the variables
        :param name: string , the name of layer
        :return: tensor , output tensor of layer
        """
    with tf.variable_scope(name_or_scope= name , reuse= reuse):

        pad0 = tf.pad(image , [[0 , 0] , [3 , 3] , [3 , 3] , [0 , 0]] , 'REFLECT')
        conv1 = relu(instance_norm(conv2d(pad0 ,gf_dim ,filter_size= 7 ,stride_size= 1 ,padding='VALID' ,name='conv1')
                                   , name= 'conv_ins_norm1'))
        conv2 = relu (instance_norm (conv2d (conv1 ,gf_dim * 2 ,filter_size=3 ,stride_size=2 ,name='conv2') ,
                                     name='conv_ins_norm2'))
        conv3 = relu (instance_norm (conv2d (conv2 ,gf_dim * 4 ,filter_size=3 ,stride_size=2 ,name='conv3') ,
                                     name='conv_ins_norm3'))

        res1 = res_block (conv3 ,gf_dim * 4 ,name='res1')
        res2 = res_block (res1 ,gf_dim * 4 ,name='res2')
        res3 = res_block (res2 ,gf_dim * 4 ,name='res3')
        res4 = res_block (res3 ,gf_dim * 4 ,name='res4')
        res5 = res_block (res4 ,gf_dim * 4 ,name='res5')
        res6 = res_block (res5 ,gf_dim * 4 ,name='res6')
        res7 = res_block (res6 ,gf_dim * 4 ,name='res7')
        res8 = res_block (res7 ,gf_dim * 4 ,name='res8')
        res9 = res_block (res8 ,gf_dim * 4 ,name='res9')

        deconv0 = deconv2d(res9 ,gf_dim * 2 ,name='deconv0')
        deconv0 = relu(instance_norm(deconv0 , 'deconv_ins_norm0'))
        deconv1 = deconv2d (deconv0 ,gf_dim ,name='deconv1')
        deconv1 = relu (instance_norm (deconv1 ,'deconv_ins_norm1'))

        # deconv0 = conv2d(upsampling(res9 , name= 'upsampling0') , gf_dim * 2 , filter_size= 3 , stride_size= 1 , name= 'conv4')
        # deconv0 = relu(instance_norm(deconv0 , 'deconv_ins_norm0'))
        # deconv1 = conv2d (upsampling (deconv0 ,name='upsampling1') ,gf_dim , filter_size=3 , stride_size=1 ,name='conv5')
        # deconv1 = relu (instance_norm (deconv1 ,'deconv_ins_norm1'))

        pad1 = tf.pad (deconv1 ,[[0 ,0] ,[3 ,3] ,[3 ,3] ,[0 ,0]] ,'REFLECT')
        pred = tanh(conv2d(pad1 , 3 , filter_size= 7 , stride_size= 1 , padding= 'VALID' , name= 'pred'))

        return pred


