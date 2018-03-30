import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class VideoEditor:
    """
    class used in editing video
    """
    def __init__(self , video_dir):
        """
        init method
        :param video_dir: string , the directory of origin video
        """
        self.video_dir = video_dir

    def transform(self ,original_name ,output_name ,fourcc ='XVID' ,output_size = (256 ,256)):
        """
        method of transform input video into target img_size video
        :param original_name: string , the name of origin video
        :param output_name: string , the name of output video
        :param fourcc: string , the fourcc of output video
        :param output_size: tuple with img_size (height , width) , output video's img_size
        :return:
        """
        self.video_capture = cv2.VideoCapture (os.path.join (self.video_dir ,original_name))
        self.fps = self.video_capture.get (cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc (*fourcc)
        self.width = self.video_capture.get (cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get (cv2.CAP_PROP_FRAME_HEIGHT)

        # if self.height >= self.width:
        #     self.width = int(self.width / (self.height / float (output_size[1])))
        #     self.width = self.width // 2 * 2
        #     self.height = output_size[1]
        # else:
        #     self.height = int(self.height / (self.width / float (output_size[0])))
        #     self.height = self.height // 2 * 2
        #     self.width = output_size[0]

        self.video_out = cv2.VideoWriter (os.path.join (self.video_dir ,output_name) , self.fourcc ,self.fps ,output_size)

        while self.video_capture.isOpened():
            ret , frame = self.video_capture.read()

            if ret:
                frame = cv2.resize (frame , (output_size[0] , output_size[1]))
                # if self.height >= self.width:
                #     frame = np.pad(frame , ((0 , 0) , ((self.height - self.width) // 2 , (self.height - self.width) // 2) , (0 , 0)) , mode= 'reflect')
                # else:
                #     frame = np.pad (frame , (((self.width - self.height) // 2 , (self.width - self.height) // 2) ,(0 , 0) , (0 , 0)), mode= 'reflect')


                self.video_out.write(frame)

            else:
                break

        self.video_capture.release()
        self.video_out.release()
        cv2.destroyAllWindows()



    def save(self ,input_name ,output_name ,sess ,faker_test_tensor ,input_tensor , fourcc ='XVID' ,
             output_size = (256 ,256) , img_channels = 3):
        """
        video using cycleGAN's transform and save method
        :param input_name:  string , the name of input video
        :param output_name:  string , the name of output video
        :param sess: Session() , model session
        :param faker_test_tensor: tensor , tensor to output the transformed frame
        :param input_tensor: placeholder , placeholder to recieve the input frame
        :param fourcc: string , the fourcc of output video
        :param output_size: tuple with img_size (height , width) , output video's img_size
        :param img_channels: int , the channels of video
        :return:
        """
        self.video_capture = cv2.VideoCapture (os.path.join (self.video_dir ,input_name))
        self.fps = self.video_capture.get (cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc (*fourcc)
        self.video_out = cv2.VideoWriter (os.path.join (self.video_dir ,output_name) ,self.fourcc ,self.fps ,output_size)



        while self.video_capture.isOpened():
            ret , frame = self.video_capture.read()
            if ret:

                out_frame = sess.run (faker_test_tensor ,feed_dict={input_tensor: [frame / 255.0]})
                out_frame = np.reshape (out_frame ,[output_size[0] ,output_size[1] ,img_channels])
                # if self.height >= self.width:
                #     out_frame = out_frame[: , self.height // 2 - self.width // 2 : self.height // 2 + self.width // 2]
                #     out_frame = np.pad(out_frame , ((0 , 0) , ((self.height - self.width) // 2 , (self.height - self.width) // 2) , (0 , 0)) , mode= 'constant')
                # else:
                #     out_frame = out_frame[self.width // 2 - self.height // 2 : self.width // 2 + self.height // 2 , :]
                #     out_frame = np.pad (out_frame , (((self.width - self.height) // 2 , (self.width - self.height) // 2) ,(0 , 0) , (0 , 0)), mode= 'constant')

                out_frame = np.uint8(out_frame * 255.0)

                self.video_out.write(out_frame)

            else:
                break

        self.video_capture.release()
        self.video_out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    input ('Video Editor will start , continue? press [Enter] to continue...')
    me = VideoEditor('video')
    me.transform('movie2.mp4' , 'in2.avi')

