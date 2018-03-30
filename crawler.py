import urllib
import os
import re
import json
import socket
import urllib.request
import urllib.parse
import urllib.error

import time

timeout = 5
socket.setdefaulttimeout(timeout)


class Crawler:
    """
    the crawler to collect images
    """
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 0

    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}

    def __init__(self , time_sleep = 0.1):
        """
        init method
        :param time_sleep: int , the time thread sleep before collect next image
        """
        self.__time_sleep = time_sleep

    def __save_image(self , response_data , dir_name):
        """
        method to save images
        :param response_data: dict  , data from Internet
        :param dir_name: string , directory of saving images
        :return:
        """

        if not os.path.exists('datasets/' + dir_name):
            os.makedirs('datasets/' + dir_name)

        self.__counter = len(os.listdir('datasets/' + dir_name)) + 1
        for image_info in response_data['imgs']:
            try:
                time.sleep(self.__time_sleep)
                fix = self.__get_suffix(image_info['objURL'])
                urllib.request.urlretrieve(image_info['objURL'] , 'datasets/' + dir_name + '/' + str(self.__counter) + str(fix))
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print('unknown error')
                continue
            else:
                print('image plus one , already have' + str(self.__counter) + 'images')
                self.__counter += 1
        return

    @staticmethod
    def __get_suffix(name):
        """
        method to get the shuffix of image
        :param name: string , the name of image
        :return: string , the suffix of the image
        """
        m = re.search(r'\.[^\.]*$' , name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    @staticmethod
    def __get_prefix(name):
        """
        the method to get prefix
        :param name: string , the name of image
        :return: the prefix of the image
        """
        return name[:name.find('.')]

    def __get_images(self ,word ='anime' , dir_name = 'data_input'):
        """
        the method to collect image from baidu
        :param word: string , the key word to search
        :param dir_name: string , the saving directory
        :return:
        """
        search = urllib.parse.quote(word)

        page_num = self.__start_amount

        while page_num < self.__amount:
            url = 'http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=' + search + '&cg=girl&pn=' +\
                  str(page_num) + '&rn=60&itg=0&z=0&fr=&width=&height=&lm=-1&ic=0&s=0&st=-1&gsm=1e0000001e'

            try:
                time.sleep(self.__time_sleep)
                req = urllib.request.Request(url= url , headers= self.headers)
                page = urllib.request.urlopen(req)
                rsp = page.read().decode('unicode_escape')

            except UnicodeDecodeError as err:
                print(err)
                print('UnicodeDecodeErrorurl:' , url)

            except urllib.error.URLError as err:
                print(err)
                print('urlErrorurl:' , url)

            except socket.timeout as err:
                print(err)
                print('socket timeout:' , url)

            else:
                response_data = json.loads(rsp)
                self.__save_image(response_data , dir_name)

                print('load next page')
                page_num += 60

            finally:
                page.close()

        print('download ends...')
        return

    def start(self , word , total_page_num , start_page = 1 , dir_name = 'data_input'):
        """
        method to start crawler
        :param word: string , the key word to seaerch
        :param total_page_num: int , the num of pages to collect
        :param start_page: int , the first page to search
        :param dir_name: string , the directory to saving results
        :return:
        """
        self.__start_amount = (start_page - 1) * 60
        self.__amount = total_page_num * 60 + self.__start_amount
        self.__get_images(word , dir_name)


if __name__ == '__main__':
    input('Crawler will start , continue? press [Enter] to continue...')
    crawler = Crawler(time_sleep= 0.1)
    crawler.start(word='狂风暴雨' ,total_page_num=50 ,start_page=1 ,dir_name='rain')
    crawler.start(word='风和日丽' ,total_page_num=50 ,start_page=1 ,dir_name='sun')

