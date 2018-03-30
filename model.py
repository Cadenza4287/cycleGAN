import tensorflow as tf
from module import *
from ops import *
from image_loader import ImageLoader
from image_pool import  ImagePool
from video_editor import VideoEditor
import os
import scipy.misc

import shutil
import datetime
import time
from glob import glob
import matplotlib.pyplot as plt

class cycleGAN:
    """
    the cycle model
    """
    def __init__(self,  sess , args):

        self.start_time = time.time ()
        self.sess = sess
        self.pool = ImagePool(max_size= args.max_size)
        self.img_size = (args.image_size , args.image_size)
        self.load_size = (args.load_size , args.load_size)
        self.img_channels = args.image_channel
        self.il = ImageLoader (load_size= self.load_size , img_size=self.img_size ,data_dir = args.data_dir ,target_dir = args.target_dir)
        self.data_dir = args.data_dir
        self.target_dir = args.target_dir
        self.video_dir = args.video_dir
        self.sample_dir = args.sample_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.output_data_dir = os.path.join ('results' ,args.output_data_dir)
        self.output_target_dir = os.path.join ('results' ,args.output_target_dir)
        self.gf_dim = args.gf_dim
        self.df_dim = args.df_dim
        self.l1_lambda = args.l1_lambda
        self.learning_rate = args.learning_rate
        self.bata1 = args.bata1
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.data_batch_num = self.il.get_image_num() // self.batch_size
        self.target_batch_num = self.il.get_image_num(is_data= False) // self.batch_size
        self.batch_num = min (self.data_batch_num ,self.target_batch_num)
        self.global_step = 0

        if args.clear_all_memory:
            print('start clear all memory...')

            def clear_files(clear_dir):
                shutil.rmtree (clear_dir)
                os.mkdir (clear_dir)

            clear_files(self.log_dir)
            clear_files(self.checkpoint_dir)
            clear_files(self.sample_dir)

            print ('successfully clear all memory...')


        if not os.path.exists('results'):
            os.makedirs('results')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._build(args)


        self.saver = tf.train.Saver()


    def _build(self , args):
        """
        build cycleGAN
        :param args: parse_args() , args from main method
        :return:
        """
        self.front_time = time.time ()
        print('building cycleGAN model...')
        with tf.device ('/device:GPU:0'):
        # get the data and target , then divide them into training part and test part
            self.data_train = self.il.load_img_batch(batch_size= self.batch_size,
                                                     num_threads= args.num_threads ,shuffle= True)
            self.data_test = self.il.load_img_batch(batch_size= self.batch_size ,
                                                    num_threads= args.num_threads , shuffle= False , is_train= False)
            self.target_train = self.il.load_img_batch(batch_size= self.batch_size,
                                                       num_threads= args.num_threads ,shuffle= True , is_data= False)
            self.target_test = self.il.load_img_batch(batch_size= self.batch_size,
                                                      num_threads= args.num_threads ,shuffle= False , is_data= False , is_train= False)

        self.coord = tf.train.Coordinator ()
        self.threads = tf.train.start_queue_runners (sess = self.sess)

        print ("got data and target...")
        #definite the training input placeholder
        with tf.device ('/device:GPU:0'):
            with tf.name_scope('input'):
                self.data_input = tf.placeholder(dtype= tf.float32 ,
                                                 shape= [None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                 name ='data_input')
                self.target_input = tf.placeholder (dtype=tf.float32 ,
                                                    shape=[None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                    name='target_input')

                #paint the input images in tensorboard
                tf.summary.image ('data_input' ,self.data_input ,1)
                tf.summary.image ('target_input' ,self.target_input ,1)

        #generative part , first use data to generate target , then use generated target to generative origin data
        with tf.device('/device:GPU:0'):
            self.faker_target = gen_resnet(self.data_input ,gf_dim= self.gf_dim ,reuse= False ,name='gen_data2target')
        with tf.device ('/device:GPU:0'):
            self.faker_data_ = gen_resnet(self.faker_target ,gf_dim= self.gf_dim , reuse= False , name= 'gen_target2data')

        with tf.device ('/device:GPU:0'):
            self.faker_data = gen_resnet(self.target_input ,gf_dim= self.gf_dim ,reuse= True ,name='gen_target2data')
            self.faker_target_ = gen_resnet(self.faker_data , gf_dim= self.gf_dim , reuse= True , name= 'gen_data2target')

        with tf.device ('/device:GPU:0'):
            #sample input means the data or target passed image pool
            with tf.name_scope('sample-input'):
                self.faker_data_input_sample = tf.placeholder(dtype= tf.float32 ,
                                                              shape= [None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                              name = 'faker_data_input_sample')
                self.faker_target_input_sample = tf.placeholder (dtype=tf.float32 ,
                                                                 shape=[None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                                 name='faker_target_input_sample')


                tf.summary.image ('faker-data-input-sample' ,self.faker_data_input_sample ,1)
                tf.summary.image ('faker-target-input-sample' ,self.faker_target_input_sample ,1)

            #calculate the probably of generated data and target
            self.faker_data_disc = disc(self.faker_data , df_dim= self.df_dim , name= 'disc_data')
            self.faker_target_disc = disc(self.faker_target , df_dim= self.df_dim , name= 'disc_target')

            #calculate the probably of origin data and target
            self.data_prob = disc(self.data_input ,df_dim = self.df_dim ,reuse= True ,name='disc_data')
            self.target_prob = disc(self.target_input ,df_dim= self.df_dim ,reuse= True ,name ='disc_target')

            # calculate the probably of generated data and target which have passed image
            self.faker_data_sample_prob = disc(self.faker_data_input_sample ,df_dim = self.df_dim ,reuse= True ,name='disc_data')
            self.faker_target_sample_prob = disc(self.faker_target_input_sample ,df_dim= self.df_dim ,reuse= True ,name ='disc_target')

        # calculate the generator loss
        with tf.device ('/device:GPU:0'):
            self.gen_loss_data2target = mean_square_loss(self.faker_target_disc , tf.ones_like(self.faker_target_disc)) + \
                                        self.l1_lambda * mean_abs_loss(self.data_input ,self.faker_data_) + \
                                        self.l1_lambda * mean_abs_loss(self.target_input ,self.faker_target_)
            self.gen_loss_target2data = mean_square_loss(self.faker_data_disc , tf.ones_like(self.faker_data_disc)) + \
                                        self.l1_lambda * mean_abs_loss(self.data_input ,self.faker_data_) + \
                                        self.l1_lambda * mean_abs_loss(self.target_input ,self.faker_target_)
            self.gen_loss = mean_square_loss(self.faker_data_disc , tf.ones_like(self.faker_data_disc)) + \
                            mean_square_loss(self.faker_target_disc , tf.ones_like(self.faker_target_disc)) + \
                            self.l1_lambda * mean_abs_loss(self.data_input ,self.faker_data_) + \
                            self.l1_lambda * mean_abs_loss(self.target_input ,self.faker_target_)

        #calculate the discriminator loss
        with tf.device ('/device:GPU:0'):
            self.disc_loss_data = mean_square_loss(self.data_prob ,tf.ones_like(self.data_prob))
            self.disc_loss_faker_data = mean_square_loss(self.faker_data_sample_prob ,tf.zeros_like(self.faker_data_sample_prob))
            self.disc_loss_data = (self.disc_loss_data + self.disc_loss_faker_data ) / 2.0

            self.disc_loss_target = mean_square_loss(self.target_prob ,tf.ones_like(self.target_prob))
            self.disc_loss_faker_target = mean_square_loss(self.faker_target_sample_prob ,tf.zeros_like(self.faker_target_sample_prob))
            self.disc_loss_target = (self.disc_loss_target + self.disc_loss_faker_target) / 2.0

        self.disc_loss = self.disc_loss_data + self.disc_loss_target


        #scalar the loss through training
        tf.summary.scalar('gen_loss_data2target' , self.gen_loss_data2target)
        tf.summary.scalar('gen_loss_target2data' , self.gen_loss_target2data)
        tf.summary.scalar('gen_loss' , self.gen_loss)

        tf.summary.scalar('disc_loss_data' , self.disc_loss_data)
        tf.summary.scalar('disc_loss_target' , self.disc_loss_target)

        #definite the test input placeholder
        with tf.name_scope('test-input'):
            self.test_data_input = tf.placeholder (dtype=tf.float32 ,
                                                   shape=[None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                   name='test_data_input')
            self.test_target_input = tf.placeholder (dtype=tf.float32 ,
                                                     shape=[None ,self.img_size[0] ,self.img_size[1] ,self.img_channels] ,
                                                     name='test_target_input')

        with tf.device ('/device:GPU:0'):
        #calculate the test generative data and target
            self.faker_test_target = gen_resnet(self.test_data_input ,gf_dim= self.gf_dim ,reuse= True ,name='gen_data2target')
            self.faker_test_data_ = gen_resnet(self.faker_test_target ,gf_dim= self.gf_dim , reuse= True , name= 'gen_target2data')

        with tf.device ('/device:GPU:0'):
            self.faker_test_data = gen_resnet(self.test_target_input ,gf_dim= self.gf_dim ,reuse= True ,name='gen_target2data')
            self.faker_test_target_ = gen_resnet(self.faker_test_data , gf_dim= self.gf_dim , reuse= True , name= 'gen_data2target')

        #get the variable list of model
        self.vars = tf.trainable_variables()

        #get generative part variable
        self.gen_vars = [v for v in self.vars if 'gen' in v.name]
        self.disc_vars = [v for v in self.vars if 'disc' in v.name]

        # tf.reset_default_graph ()

        print("sussessfully init variable")


        print ('total build time: %02f sec...' % (time.time () - self.front_time))
        print('successfully build cycleGAN model...')



    def train(self , args):

        self.current_learning_rate = tf.placeholder (dtype=tf.float32 ,name='current_learning_rate')
        # definate the optimizer to minimize the discriminate loss and generative loss
        with tf.device ('/device:GPU:0'):
            self.disc_optim = tf.train.AdamOptimizer (learning_rate=self.current_learning_rate ,beta1=self.bata1). \
                minimize (self.disc_loss ,var_list=self.disc_vars)
        with tf.device ('/device:GPU:0'):
            self.gen_optim = tf.train.AdamOptimizer (learning_rate=self.current_learning_rate ,beta1=self.bata1). \
                minimize (self.gen_loss ,var_list=self.gen_vars)

        self.merged = tf.summary.merge_all ()
        self.writer = tf.summary.FileWriter (
            os.path.join (self.log_dir ,datetime.datetime.now ().strftime ("%Y%m%d-%H%M%S")) ,
            self.sess.graph)

        #init the variables of model
        self.sess.run (tf.global_variables_initializer ())
        self.writer = tf.summary.FileWriter (self.log_dir ,self.sess.graph)

        # whether to use checkpoint to support continuous training
        if args.continue_train:
            if self._load(self.checkpoint_dir):
                print('success to load checkpoint')
            else:
                print('fail to load checkpoint')

        print ('start training...')
        #do training epoch
        try:
            for epoch in range(self.global_step // self.batch_num , self.epoch_num):
                # dataA = glob ('./datasets/' + self.data_dir + '_train/%sx%s/*.*' %(self.img_size[0] , self.img_size[1]))
                # dataB = glob ('./datasets/' + self.target_dir + '_train/%sx%s/*.*' %(self.img_size[0] , self.img_size[1]))
                # np.random.shuffle (dataA)
                # np.random.shuffle (dataB)
                for b in range(self.batch_num):
                    # data_batch = list(dataA[b * self.batch_size:(b + 1) * self.batch_size])
                    # target_batch = list(dataB[b * self.batch_size:(b + 1) * self.batch_size])
                    # data_batch = [load_train_data (batch_file ,args.load_size ,args.image_size) for batch_file in
                    #                 data_batch]
                    # target_batch = [load_train_data (batch_file ,args.load_size ,args.image_size) for batch_file in
                    #                 target_batch]
                    # data_batch = np.array (data_batch).astype (np.float32)
                    # target_batch = np.array (target_batch).astype (np.float32)
                    step = epoch * self.batch_num + b
                    # decay learning rate at the last half training process
                    learning_rate = self.learning_rate if epoch < args.epoch_decay \
                        else (self.epoch_num - epoch) / (self.epoch_num - args.epoch_decay) * self.learning_rate
                    #run optimizer process
                    data_batch ,target_batch = self.sess.run([self.data_train ,self.target_train])
                    self.sess.run(self.gen_optim ,feed_dict={self.data_input:data_batch ,
                                                             self.target_input : target_batch ,
                                                             self.current_learning_rate: learning_rate})
                    temp_data ,temp_target = self.sess.run([self.faker_data , self.faker_target],
                                                           feed_dict={self.data_input:data_batch ,self.target_input : target_batch})
                    temp_data ,temp_target = self.pool([temp_data , temp_target])
                    self.sess.run(self.disc_optim ,feed_dict= {self.data_input: data_batch ,self.target_input: target_batch ,
                                                               self.faker_data_input_sample:temp_data ,
                                                               self.faker_target_input_sample: temp_target ,
                                                               self.current_learning_rate: learning_rate})

                    #per merged_frequent times training write into tensorboard
                    if (step + 1) % args.merged_frequent == 0:
                        l1 , l2 , summary = self.sess.run([self.gen_loss , self.disc_loss , self.merged] ,
                                                          feed_dict= {self.data_input: data_batch ,self.target_input: target_batch ,
                                                                      self.faker_data_input_sample:temp_data ,
                                                                      self.faker_target_input_sample: temp_target})
                        self.writer.add_summary (summary ,step)
                        print('global step %s -- epoch %s : gen loss: %s disc loss: %s current learning rate %.06f'
                              %(step + 1 , epoch , l1 , l2 , learning_rate))

                    # per save_frequent times training save a datapoint
                    if (step + 1) % args.save_frequent == 0:
                        self._save(self.checkpoint_dir , step + 1)

                    # per sample_frequent times training sample the results of test examples
                    if (step + 1) % args.sample_frequent == 0:
                        self._sample(args , step + 1)
        except Exception as e:
            print(e)
            self.coord.request_stop (e)

        self.coord.request_stop()
        self.coord.join(self.threads)

        print ('successfully end training...')

    def _save(self ,checkpoint_dir ,step):
        """
        the method of save checkpoints
        :param checkpoint_dir: string , the directory of saving checkpoint
        :param step: int , the global training step
        :return:
        """
        print('saving model...')
        model_name = 'cyclegan.model'
        model_dir = '%s_%s' %(self.data_dir , self.img_size)
        checkpoint_dir = os.path.join (checkpoint_dir ,model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess , os.path.join(checkpoint_dir , model_name) , global_step= step)
        print('successfully saving model...')


    def _load(self ,checkpoint_dir):
        """
        method of load checkpoint
        :param checkpoint_dir: string , the directory of saving checkpoint
        :return: bool , whether load success
        """
        print('reading checkpoint...')

        model_dir = '%s_%s' %(self.data_dir , self.img_size)
        checkpoint_dir = os.path.join(checkpoint_dir , model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.global_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess , os.path.join(checkpoint_dir , ckpt_name))
            return True
        else:
            return False

    def save_result(self , result ,output_dir ,name):
        """
        method of save result image
        :param result:
        :param output_dir:
        :param name:
        :return:
        """
        result = np.reshape (result ,[self.img_size[0] ,self.img_size[1] ,self.img_channels])
        result = ((result + 1.0) * 127.5).astype (int)
        scipy.misc.imsave (os.path.join (output_dir ,name + '.jpg') ,result)

    def _sample(self , args , step):
        """
        method of sample images
        :param args: arg_parse() , args from main
        :param step: int , the global training step
        :return:
        """
        print('global step %03d -- sample images ...' %step)
        if not os.path.exists (args.sample_dir):
            os.makedirs (args.sample_dir)

        iter_dir = os.path.join(args.sample_dir , 'step-%06d' %step)

        if not os.path.exists (iter_dir):
            os.makedirs (iter_dir)

        test_num = self.il.get_image_num (is_data=True ,is_train=False)
        for i in range (min(test_num , args.sample_num)):
            data_test_batch = self.sess.run (self.data_test)

            original_data ,result_target ,result_data_ = \
                self.sess.run ([self.test_data_input ,self.faker_test_target ,self.faker_test_data_] ,
                               feed_dict={self.test_data_input: data_test_batch})

            self.save_result (original_data ,iter_dir ,'%03d' % i + 'data-original')
            self.save_result (result_target ,iter_dir ,'%03d' % i + 'target-transformed')
            self.save_result (result_data_ ,iter_dir ,'%03d' % i + 'data-rebuild')

        test_num = self.il.get_image_num (is_data=False ,is_train=False)
        for i in range (min(test_num , args.sample_num)):
            target_test_batch = self.sess.run (self.target_test)
            original_target ,result_data ,result_target_ = \
                self.sess.run ([self.test_target_input ,self.faker_test_data ,self.faker_test_target_] ,
                               feed_dict={self.test_target_input: target_test_batch})

            self.save_result (original_target ,iter_dir ,'%03d' % i + 'target-original')
            self.save_result (result_data ,iter_dir ,'%03d' % i + 'data-transformed')
            self.save_result (result_target_ ,iter_dir ,'%03d' % i + 'target-rebuild')

        print('global step %03d -- successfully sample images ...' %step)


    def test(self , args):
        """
        method of test
        :param args:args: arg_parse() , args from main
        :return:
        """
        print("start testing...")
        self.front_time = time.time ()
        self.sess.run (tf.global_variables_initializer ())

        if self._load (self.checkpoint_dir):
            print ('success to load checkpoint')
        else:
            print ('fail to load checkpoint')

        if args.image_transform:
            print('test image...')
            if not os.path.exists (self.output_data_dir):
                os.makedirs (self.output_data_dir)

            if not os.path.exists (self.output_target_dir):
                os.makedirs (self.output_target_dir)

            if args.test_direction_data2target:
                test_num = self.il.get_image_num(is_data= True , is_train= False)
                for i in range(test_num):
                    data_test_batch = self.sess.run(self.data_test)

                    original_data , result_target , result_data_ = \
                        self.sess.run([self.test_data_input , self.faker_test_target , self.faker_test_data_] ,
                                                                 feed_dict={self.test_data_input:data_test_batch})

                    self.save_result(original_data , self.output_target_dir , '%03d' % i + '-original')
                    self.save_result(result_target , self.output_target_dir , '%03d' % i + '-transformed')
                    self.save_result(result_data_ , self.output_target_dir , '%03d' % i + '-rebuild')
                    # result_target = np.reshape (result_target ,[self.img_size[0] ,self.img_size[1] ,self.img_channels])
                    # result_data_ = np.reshape (result_data_ ,[self.img_size[0] ,self.img_size[1] ,self.img_channels])
                    # result_target = (result_target * 255.0).astype(int)
                    # result_data_ = (result_data_ * 255.0).astype(int)
                    # scipy.misc.imsave (os.path.join (self.output_target_dir ,str (i) + '.jpg') ,result_target)
                    # scipy.misc.imsave (os.path.join (self.output_target_dir ,str (i) + '.jpg') ,result_data_)
            else:

                test_num = self.il.get_image_num (is_data= False ,is_train=False)
                for i in range(test_num):
                    target_test_batch = self.sess.run(self.target_test)
                    original_target , result_data , result_target_ = \
                        self.sess.run([self.test_target_input , self.faker_test_data , self.faker_test_target_] ,
                                                                 feed_dict={self.test_target_input:target_test_batch})

                    self.save_result (original_target ,self.output_data_dir ,'%03d' % i + '-original')
                    self.save_result (result_data ,self.output_data_dir ,'%03d' % i + '-transformed')
                    self.save_result (result_target_ ,self.output_data_dir ,'%03d' % i + '-rebuild')
                    # result_data = np.reshape(result_data , [self.img_size[0] , self.img_size[1] , self.img_channels])
                    # result_target_ = np.reshape(result_target_ , [self.img_size[0] , self.img_size[1] , self.img_channels])
                    # result_data = (result_data * 255.0).astype(int)
                    # result_target_ = (result_target_ * 255.0).astype(int)
                    # scipy.misc.imsave (os.path.join (self.output_target_dir ,str (i) + '.jpg') ,result_target_)
            print('successfully test images...')

        else:
            print('test video...')
            self.video_editor = VideoEditor (video_dir=args.video_dir)
            self.video_editor.transform(args.original_video , args.input_video)

            if args.test_direction_data2target:
                self.video_editor.save(input_name = args.input_video ,output_name= args.output_video ,
                                       sess= self.sess ,faker_test_tensor= self.faker_test_target ,
                                       input_tensor= self.test_data_input)
            else:
                self.video_editor.save (input_name=args.input_video ,output_name=args.output_video ,
                                        sess= self.sess , faker_test_tensor= self.faker_test_data ,
                                        input_tensor= self.test_target_input)

            print ('total test time: %02f sec' % (time.time () - self.front_time))
            print('successfully test video...')


