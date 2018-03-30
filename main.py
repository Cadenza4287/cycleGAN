import argparse
import tensorflow as tf
import os

from model import cycleGAN

paser = argparse.ArgumentParser(description='')

paser.add_argument('--data_dir' , dest= 'data_dir' , default= 'reality' , help='path of data_input images')
paser.add_argument('--target_dir' , dest= 'target_dir' , default= 'oils' , help='path of target_input images')
paser.add_argument('--log_dir' , dest= 'log_dir' , default= 'log' , help='path of logger')
paser.add_argument('--sample_dir' , dest= 'sample_dir' , default= 'sample' , help='path of sample images')
paser.add_argument('--output_data_dir' , dest= 'output_data_dir' , default= 'reality-result' , help='path of data output images')
paser.add_argument('--output_target_dir' , dest= 'output_target_dir' , default= 'oils-result' , help='path of target output images')
paser.add_argument('--checkpoint_dir' , dest= 'checkpoint_dir' , default= 'checkpoint' , help='path of check point')
paser.add_argument('--video_dir' , dest= 'video_dir' , default= 'video' , help='path of input and output video')
paser.add_argument('--input_video' , dest= 'input_video' , default= 'in3.avi' , help='the name of input video')
paser.add_argument('--output_video' , dest= 'output_video' , default= 'out3.avi' , help='the name of output video')
paser.add_argument('--original_video' , dest= 'original_video' , default= 'movie3.mp4' , help='the name of original video')
paser.add_argument('--max_size' , dest= 'max_size' , type= int , default= 50 , help='img_size of image pool')
paser.add_argument('--gf_dim' , dest= 'gf_dim' , type= int , default= 64 , help='# of generator first conv hidden units')
paser.add_argument('--df_dim' , dest= 'df_dim' , type= int , default= 64 , help='# of discriminator first conv hidden units')
paser.add_argument('--l1_lambda' , dest= 'l1_lambda' , type= float , default= 10.0 , help='weight on L1 term in objective')
paser.add_argument('--learning_rate' , dest= 'learning_rate' , type= float , default= 0.0002 , help='learning rate of AdamOptimizer')
paser.add_argument('--bata1' , dest= 'bata1' , type= float , default= 0.5 , help='bata1 of AdamOptimizer')
paser.add_argument('--load_size' , dest= 'load_size' , type= int , default= 286 , help='img_size of load')
paser.add_argument('--image_size' , dest= 'image_size' , type= int , default= 256 , help='img_size of image')
paser.add_argument('--image_channel' , dest= 'image_channel' , type= int , default= 3 , help='channels of image')
paser.add_argument('--epoch_num' , dest= 'epoch_num' , type= int , default= 200 , help='# of training epoch')
paser.add_argument('--batch_size' , dest= 'batch_size' , type= int , default= 1 , help='# of examples per batch')
# paser.add_argument('--capacity' , dest= 'capacity' , type= int , default= 100 , help='# of capacity in image loader')
# paser.add_argument('--min_after_dequeue' , dest= 'min_after_dequeue' , type= int , default= 10 ,
#                                                                                 help='# of min_after_dequeue in image loader')
paser.add_argument('--num_threads' , dest= 'num_threads' , type= int , default= 4 , help='# of threads in image loader')
paser.add_argument('--continue_train' , dest= 'continue_train' , type= bool , default= True , help='whether continue training')
paser.add_argument('--save_frequent' , dest= 'save_frequent' , type= int , default= 1000 , help='the frequent of saving checkpoint')
paser.add_argument('--merged_frequent' , dest= 'merged_frequent' , type= int , default= 100 , help='the frequent of write merged')
paser.add_argument('--sample_frequent' , dest= 'sample_frequent' , type= int , default= 4000 , help='the frequent of sample images')
paser.add_argument('--sample_num' , dest= 'sample_num' , type= int , default= 50 , help='the frequent of sample images')
paser.add_argument('--epoch_decay' , dest= 'epoch_decay' , type= int , default= 100 , help='# of step to start decay learning rate')
paser.add_argument('--is_train' , dest= 'is_train' , type= bool , default=True , help='whether is training')
paser.add_argument('--clear_all_memory' , dest= 'clear_all_memory' , type= bool , default= False , help='whether clear the memory ')
paser.add_argument('--test_direction_data2target' , dest= 'test_direction_data2target' , type= bool , default= True , help='the direction of transform '
                                                                                                   'True means data to target , '
                                                                                                   'False means target to data')
paser.add_argument('--image_transform' , dest= 'image_transform' , type= bool , default= True , help='True means image transform'
                                                                                                     'False means video transform')

args = paser.parse_args()

def main(_):
    """
    global main method
    """
    # input ('Main method will start , continue? press [Enter] to continue...')

    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.InteractiveSession(config=tfconfig)
    gan = cycleGAN (sess , args)
    if args.is_train:
        gan.train (args)
    else:
        gan.test (args)

if __name__ == '__main__':
    tf.app.run()


