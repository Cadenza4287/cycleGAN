import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.misc
from PIL import Image


class ImageLoader:
    """
    include the method to load images
    """
    __dataset_dir = 'datasets'
    __data_train_dir = ''
    __data_test_dir = ''

    __target_train_dir = ''
    __target_test_dir = ''

    img_size = None
    channels = 3

    def __init__(self , load_size , img_size ,data_dir ,target_dir):
        """
        init method
        :param data_dir: the directory of data
        :param target_dir: the directory of target
        """
        self.__data_train_dir = os.path.join(self.__dataset_dir , data_dir + '_train')
        self.__data_test_dir = os.path.join(self.__dataset_dir , data_dir + '_test')
        self.__target_train_dir = os.path.join(self.__dataset_dir , target_dir + '_train')
        self.__target_test_dir = os.path.join(self.__dataset_dir , target_dir + '_test')

        self.load_size = load_size
        self.img_size = img_size

    def __read_img(self , filename_quene):
        """
        images reader method
        :param filename_quene: the quene of files
        :return:
        """
        reader = tf.WholeFileReader()
        key , value = reader.read(filename_quene)
        record_bytes = tf.image.decode_image(value , channels= 3)

        return record_bytes


    def __load_img(self , is_data = True , is_train = True ,  shuffle = True):
        """
        method to load images
        :param is_data: whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :param is_train: whether to load images in training folder
        :param shuffle: whether to shuffle load images
        :return:
        """

        if is_data:
            if is_train:
                file_dir = self.__data_train_dir
            else:
                file_dir = self.__data_test_dir

        else:
            if is_train:
                file_dir = self.__target_train_dir
            else:
                file_dir = self.__target_test_dir

        file_dir = os.path.join (file_dir ,'%sx%s' % (self.img_size[0] ,self.img_size[1]))

        file_name = os.listdir(file_dir)

        file_name = [os.path.join (file_dir ,f) for f in file_name]
        filename_queue = tf.train.string_input_producer(file_name , shuffle= shuffle)
        read_input = self.__read_img(filename_queue)

        read_input = tf.reshape (read_input ,[self.img_size[0] ,self.img_size[1] ,self.channels])

        if is_train:
            img = tf.image.resize_images(read_input , [self.load_size[0] , self.load_size[1]])
            h1 = int(np.ceil(np.random.uniform(1e-2, self.load_size[0]-self.img_size[0])))
            w1 = int(np.ceil(np.random.uniform(1e-2, self.load_size[1]-self.img_size[1])))
            img = tf.image.crop_to_bounding_box(img , h1 , w1 , self.img_size[0] , self.img_size[1])
            img = tf.image.random_flip_left_right(img)
        else:
            img = tf.image.resize_images(read_input , (self.img_size[0] , self.img_size[1]))

        reshaped_img = tf.subtract(tf.divide(tf.cast(img , tf.float32) ,tf.constant(127.5)) ,tf.constant(1.0))

        return reshaped_img

    def load_img_batch(self , batch_size , num_threads ,
                       is_data = True , is_train = True ,  shuffle = True):
        """
        method to load batch images
        :param batch_size: the img_size of image batch
        :param num_threads: The number of threads enqueuing
        :param is_data: whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :param is_train: whether to load images in training folder
        :param shuffle: whether to shuffle load images
        :return:
        """
        image = self.__load_img (is_data= is_data ,is_train = is_train ,  shuffle = shuffle)
        img_num = self.get_image_num (is_data ,is_train)
        min_dequeue_examples = int (0.1 * img_num)
        __img_batch = tf.train.batch([image] ,batch_size = batch_size ,
                                     capacity= 50 ,
                                     num_threads= num_threads )
        return __img_batch

    # def start_queue(self , sess):
    #     """
    #     before load image batch , you need to run this method
    #     :param sess: the Session of tensorflow
    #     :return:
    #     """
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners (sess=sess)
    #
    #     return coord , threads

    # def get_size(self):
    #     """
    #     get the img_size of output image
    #     :return: tuple with img_size height * width , img_size of output image
    #     """
    #     return self.img_size

    # def get_channels(self):
    #     """
    #     get the channels of iamges
    #     :return: int , channels of images
    #     """
    #     return self.channels
    #
    def get_image_num(self , is_data = True , is_train = True):
        """
        get the number of images in folder
        :param is_data: whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :param is_train: whether to load images in training folder
        :return: int , the num of images
        """
        if is_data:
            if is_train:
                file_dir = self.__data_train_dir
            else:
                file_dir = self.__data_test_dir

        else:
            if is_train:
                file_dir = self.__target_train_dir
            else:
                file_dir = self.__target_test_dir

        file_dir = os.path.join (file_dir ,'%sx%s' % (self.img_size[0] ,self.img_size[1]))

        return len(os.listdir(file_dir))

# if __name__ == '__main__':
#     sess = tf.Session()
#     il = ImageLoader (data_dir='reality' ,target_dir='anime')
#     image = il.load_img ()
#     img_batch = tf.train.shuffle_batch ([image] ,batch_size=3 ,capacity=200 ,min_after_dequeue=100 ,num_threads=2)
#     sess.run (tf.global_variables_initializer ())
#     tf.train.start_queue_runners (sess=sess)
#
#     for i in range(2):
#         image_value = sess.run(img_batch)
#         print(image_value)
#         plt.imshow(image_value[0])
#         plt.show()

