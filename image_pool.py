import numpy as np
import copy

class ImagePool:
    """
        image pool is used in remembering the knowledge that the model learned before
        when model meets a new example , that example has 50 percent probably to be changed with an example meet before
    """
    def __init__(self , max_size):
        """
        init method
        :param max_size: int , the img_size of image pool
        """
        self.max_size = max_size
        self.num_img = 0
        self.images = []

    def __call__(self, images):
        """
        when image pool called , do the example-change process
        :param images: list of shape [2 , height , width , channels] , input images
        :return: the same shape with input images , output images
        """
        if self.max_size <=0:
            return images

        if self.num_img <self.max_size:
            self.images.append(images)
            self.num_img +=1
            return images

        if np.random.rand() > 0.5:
            idx1 = int(np.random.rand() * self.max_size)
            img1 = copy.copy(self.images[idx1][0])
            self.images[idx1][0] = images[0]

            idx2 = int(np.random.rand () * self.max_size)
            img2 = copy.copy (self.images[idx2][1])
            self.images[idx2][1] = images[1]

            return [img1 , img2]

        else:
            return images

