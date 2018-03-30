from PIL import Image
import os
import shutil
import numpy as np

class ImageEditor:
    """
    include the method to transform image type, resize image , mirror image and shuffed_separate
    """
    # __data_dir = ''
    # __data_train_dir = ''
    # __data_test_dir = ''
    #
    # __target_dir = ''
    # __target_train_dir = ''
    # __target_test_dir = ''

    def __init__(self , target_size  , data_dir = 'reality' , target_dir = 'anime'):
        """
        init method
        :param data_dir: string , the directory of data
        :param target_dir: string , the directory of target
        """
        self.__data_dir = os.path.join('datasets', data_dir)
        self.__data_train_dir = self.__data_dir + '_train'
        self.__data_test_dir = self.__data_dir + '_test'
        self.__data_pixel_dir = self.__data_dir + '_pixel'

        self.__target_dir = os.path.join('datasets', target_dir)
        self.__target_train_dir = self.__target_dir + '_train'
        self.__target_test_dir = self.__target_dir + '_test'
        self.__target_pixel_dir = self.__target_dir + '_pixel'

        self.__target_size = target_size
        self.__pixel_folder = ["%sx%s" %(i , i) for i in self.__target_size]

    def makedirs(self):
        """
        method to make needed directory
        """
        dir_dic = [self.__data_train_dir , self.__data_test_dir , self.__target_train_dir , self.__target_test_dir]
        print ('make dirs start...')
        for d in dir_dic:
            if not os.path.exists(d):
                os.makedirs(d)

            for f in self.__pixel_folder:
                if not os.path.exists(os.path.join(d , f)):
                    os.makedirs(os.path.join(d , f))
                if not os.path.exists(os.path.join(d , f)):
                    os.makedirs(os.path.join(d , f))



    def transform_type(self , output_type = 'jpg' , is_data = True):
        """

        :param output_type: string , output type
        :param is_data: bool , whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :return:
        """
        if is_data:
            file_dir = self.__data_dir
            print ('transform data_input image start ...')
        else:
            file_dir = self.__target_dir
            print ('transform target_input image start...')

        data_name = os.listdir(file_dir)
        for id , name in enumerate(data_name):
            out_name = str (file_dir.split ('\\')[-1]) + '%04d' % id + '.' + output_type

            try:
                img = Image.open(os.path.join(file_dir , name)).convert('RGB')
                os.remove (os.path.join (file_dir ,name))
                img.save (os.path.join (file_dir ,out_name))

            except IOError as err:
                print('can not convert' + os.path.join(file_dir , name))
                print(err)
                if os.path.exists(os.path.join(file_dir , out_name)):
                    os.remove(os.path.join(file_dir , out_name))
                    os.remove (os.path.join (file_dir ,name))

    def crop_separate_img(self , height_crop , width_crop , test_percent = 0.1 ,  is_data = True):
        """
        method to crop image
        :param height_crop: int , proportion to crop images' height , to crop subtitile and copyright
        :param width_crop: int , proportion to crop images' width, to crop subtitile and copyright
        :param target_size: list with shape [size1 , size2...] final output img_size of image
        :param is_data: bool , whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :return:
        """
        if is_data:
            file_dir = self.__data_dir
            file_train_dir = self.__data_train_dir
            file_test_dir = self.__data_test_dir
            print ('crop and separate data_input image start...')
        else:
            file_dir = self.__target_dir
            file_train_dir = self.__target_train_dir
            file_test_dir = self.__target_test_dir
            print ('crop and separate target_input image start...')

        data_name = os.listdir (file_dir)
        num_data = len (data_name)
        train_indices = np.random.choice (range (num_data) ,int (num_data * (1.0 - test_percent)) ,
                                          replace=False)

        for idn , name in enumerate(data_name):
            term1 ,term2 = os.path.splitext (name)
            try:
                img = Image.open(os.path.join(file_dir , name))
                size = img.size
                if size[0] < size[1]:
                    height = size[0] * height_crop
                    width = size[0] * width_crop
                else:
                    height = size[1] * height_crop
                    width = size[1] * width_crop
                height = size[1] * height_crop
                width = size[0] * width_crop
                box = ((size[0] - width) / 2 , (size[1] - height) / 2 ,
                       size[0] - (size[0] - width) / 2 , size[1] - (size[1] - height) / 2)
                img_box = img.crop(box)
                for id , t in enumerate(self.__target_size):
                    img_resize = img_box.resize((t , t) , resample = Image.BICUBIC)
                    if idn in train_indices:
                        img_resize.save(os.path.join(file_train_dir , self.__pixel_folder[id] ,term1) + term2)
                    else:
                        img_resize.save (os.path.join (file_test_dir ,self.__pixel_folder[id] ,term1) + term2)

                # os.remove(os.path.join(file_dir , name))

            except IOError as err:
                print('can not crop ' + name)
                print(err)
                continue

    # def mirror_img(self , is_data = True):
    #     """
    #     method to mirror the img , create more examples
    #     :param is_data: bool , whether to transform type in data folder , True means transform type in data folder ,
    #                     else in target folder
    #     :return:
    #     """
    #     if is_data:
    #         file_dir = self.__data_dir
    #         print ('mirror data_input image start...')
    #     else:
    #         file_dir = self.__target_dir
    #         print ('mirror target_input image start...')
    #
    #     data_name = os.listdir(file_dir)
    #
    #     for name in data_name:
    #         term1 , term2 = os.path.splitext(name)
    #         if not name.endswith('m' + term2):
    #             try:
    #                 img= Image.open(os.path.join(file_dir , name))
    #                 img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
    #                 img_mirror.save(os.path.join(file_dir ,term1) + 'm' + term2)
    #
    #             except IOError as err:
    #                 print(err)
    #                 print('can not mirror' + name)
    #                 continue



    def shuffed_separate(self , test_percent = 0.2 , is_data = True):
        """
        method to separate train and test examples
        :param test_percent: float in (0 , 1) , the proportion of test examples
        :param is_data:bool , whether to transform type in data folder , True means transform type in data folder ,
                        else in target folder
        :return:
        """
        if is_data:
            file_dir = self.__data_dir
            file_train_dir = self.__data_train_dir
            file_test_dir = self.__data_test_dir
            print ('shuffed separate data_input image start...')
        else:
            file_dir = self.__target_dir
            file_train_dir = self.__target_train_dir
            file_test_dir = self.__target_test_dir
            print ('shuffed sparate target_input image start...')

        data_name = os.listdir(file_dir)
        num_data = len(data_name)
        train_indices = np.random.choice(range(int(num_data / 2)) , int(num_data / 2 * (1.0 - test_percent)) ,
                                              replace= False)
        # test_indices = list(set(range(int(num_data / 2))) - set(train_indices))

        for id , name in enumerate(data_name):
            try:
                if id // 2 in train_indices:
                    shutil.move(os.path.join(file_dir , name) , os.path.join(file_train_dir , name))
                else:
                    shutil.move (os.path.join (file_dir ,name) ,os.path.join (file_test_dir ,name))

            except IOError as err:
                print(err)
                print('can not move' + name)

    def deletedirs(self):
        try:
            os.removedirs(self.__data_dir)
            os.removedirs(self.__target_dir)

        except IOError as err:
            print(err)

    def pixel_separate(self , size = None  ,is_data = True):
        if size is None:
            size = [4 ,8 ,16 ,32 ,64 ,128 ,256 , 512 , 1024]
        if is_data:
            file_dir = self.__data_dir
            file_train_dir = self.__data_train_dir
            file_test_dir = self.__data_test_dir
            file_pixel_dir = self.__data_pixel_dir
            print ('pixel_separate data_input image start...')

        else:
            file_dir = self.__target_dir
            file_train_dir = self.__target_train_dir
            file_test_dir = self.__target_test_dir
            file_pixel_dir = self.__target_pixel_dir
            print ('pixel_separate target_input image start...')

        file_name = os.listdir(file_train_dir)

        for i in size:
            if not os.path.exists (os.path.join (file_pixel_dir ,'%sx%s' %(i , i))):
                os.makedirs(os.path.join (file_pixel_dir ,'%sx%s' %(i , i)))
            print("resize to %s x %s start..." %(i , i))
            for j in file_name:
                # print ('%s image start...' % j)
                img = Image.open (os.path.join(file_train_dir, j))
                img_resize = img.resize((i , i))
                img_resize.save(os.path.join(file_pixel_dir , '%sx%s' %(i , i) , j))
                # print ('%s image succeed...' % j)
            print ("resize to %s x %s end..." %(i , i))

if __name__ == '__main__':
    input ('image editor will start , continue? press [Enter] to continue...')
    ie = ImageEditor(target_size= [256] , data_dir= 'reality' , target_dir= 'oils')
    ie.makedirs()
    # ie.transform_type ()
    # ie.transform_type (is_data= False)
    ie.crop_separate_img(height_crop= 0.8 , width_crop= 0.8)
    ie.crop_separate_img (height_crop=0.8 ,width_crop=0.8 , is_data= False)
    # # ie.mirror_img()
    # # ie.mirror_img(is_data= False)
    # ie.shuffed_separate(test_percent= 0.02)
    # ie.shuffed_separate(test_percent= 0.1 , is_data= False)
    # # ie.deletedirs()
    # ie.pixel_separate(is_data= False)