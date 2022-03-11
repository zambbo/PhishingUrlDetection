import json
import os
from config import *
import random
# json file을 불러온다
# return : dict 
def loadJson(filename):
    ret_json = None
    try:
        with open(filename, "r") as json_file:
            ret_json = json.load(json_file)
    except Exception as e:
        print(f"Error while opening {filename}.\nerror message: {e}")
    return ret_json

#json file로부터 읽어온다. 
def getUrls(json_dict, url_type:int = 0):

    if url_type <0 or url_type >1: raise Exception("Invalid Parameter(url_type) Error!")

    if url_type == 0:
        ret_url_list = [obj['url'] for obj in json_dict]

    if url_type == 1:
        ret_url_list = [obj['url'] for obj in json_dict['_embedded']['phish']]


    return ret_url_list

# input : filepath
# output : url list
def getUrls_f(_filenames, url_type: int = 0):
    ret_url_list = []
    if type(_filenames) == str:
            
        json_dict = loadJson(_filenames)

        ret_url_list = getUrls(json_dict)

    if type(_filenames) == list:
        for filename in _filenames:
            json_dict = loadJson(filename)

            ret_url_list.extend(getUrls(json_dict))

             
    return ret_url_list

# url_type = 0 일경우 benign 데이터, 1일 경우 phishing 데이터 가져옴
# 
def getFilePaths(dir_path, url_type:int = 0, **kwargs):

    shuffle = False
    path_list = []

    if 'shuffle' in kwargs.keys():
        shuffle = kwargs[shuffle]

    if url_type <0 or url_type >1: raise Exception("Invalid Parameter(url_type) Error!")

    #benign data
    if url_type == 0:

    #phishing data
    if os.path.isdir(dir_path):
        pass

    # 만약 shuffle 옵션이 켜져있다면 랜덤으로 셔플한다.
    if shuffle: random.shuffle(path_list)
