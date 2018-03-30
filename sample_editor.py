import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class SampleEditor:

    def __init__(self , sample_dir , result_dir , output_name = 'result.avi' , test_direction_data2target = True):
        self.sample_dir = sample_dir
        self.result_dir = result_dir
        self.video_height = 720
        self.video_width = 1280
        self.image_num = 6
        self.fps = 6
        self.delta_time = 1
        self.fourcc = cv2.VideoWriter_fourcc (*'XVID')
        self.test_direction_data2target = test_direction_data2target
        self.video_out = cv2.VideoWriter (os.path.join (self.result_dir ,output_name) ,self.fourcc ,self.fps ,
                                          (self.video_width , self.video_height))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create(self):
        self.sample_list = [os.path.join(self.sample_dir , i) for i in os.listdir(self.sample_dir)]
        self.sample_num = len(self.sample_list)
        for i in range(self.image_num):
            for j in self.sample_list:
                background = np.zeros ((self.video_height ,self.video_width ,3) ,np.uint8)
                image_list = [os.path.join(j , l) for l in os.listdir (j)]
                image_original = cv2.imread(image_list[i * 6])
                image_transform = cv2.imread(image_list[i * 6 + 5])
                image_rebuild = cv2.imread(image_list[i * 6 + 1])
                image_height = np.size(image_original , 0)
                image_width = np.size(image_original , 1)
                background[self.video_height // 2 - image_height // 2: self.video_height // 2 + image_height // 2 ,
                self.video_width // 4 - image_width // 2 : self.video_width // 4 + image_width // 2 , :] = image_original
                cv2.putText (background ,'    original' ,
                             ( self.video_width // 4 - image_width // 2 , self.video_height // 2 + image_height // 3 * 2) ,
                             self.font ,1 ,(255 ,255 ,255) , 2 , cv2.LINE_AA , bottomLeftOrigin= False)
                background[self.video_height // 2 - image_height // 2: self.video_height // 2 + image_height // 2 ,
                self.video_width // 4 * 2 - image_width // 2 :self.video_width // 4 * 2 + image_width // 2 , :] = image_transform
                cv2.putText (background ,'    transform' ,
                             (self.video_width // 4 * 2 - image_width // 2 ,self.video_height // 2 + image_height // 3 * 2) ,
                             self.font ,1 ,(255 ,255 ,255) ,2 ,cv2.LINE_AA ,bottomLeftOrigin=False)
                background[self.video_height // 2 - image_height // 2: self.video_height // 2 + image_height // 2 ,
                self.video_width // 4 * 3 - image_width // 2 :self.video_width // 4 * 3 + image_width // 2 , :] = image_rebuild
                cv2.putText (background ,'    rebuild' ,
                             (self.video_width // 4 * 3 - image_width // 2 ,self.video_height // 2 + image_height // 3 * 2) ,
                             self.font ,1 ,(255 ,255 ,255) ,2 ,cv2.LINE_AA ,bottomLeftOrigin=False)

                # cv2.putText (background ,'OpenCV' ,(10 ,500) ,font ,4 ,(255 ,255 ,255) ,2 ,cv2.LINE_AA)


                self.video_out.write(background)


            background = np.zeros ((self.video_height ,self.video_width ,3) ,np.uint8)
            for j in range(self.delta_time * self.fps):
                self.video_out.write (background)

        self.video_out.release ()
        cv2.destroyAllWindows ()



if __name__ == '__main__':
    input ('Sample Editor will start , continue? press [Enter] to continue...')
    se = SampleEditor('sample' , 'results')
    se.create()


