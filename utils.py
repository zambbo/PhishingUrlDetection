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
def getUrls(json_dict, url_type:int = 0, max_len:int = -1):

    if url_type <0 or url_type >1: raise Exception("Invalid Parameter(url_type) Error!")

    if url_type == 0:
        ret_url_list = [obj['url'] for obj in json_dict]

    if url_type == 1:
        ret_url_list = [obj['url'] for obj in json_dict['_embedded']['phish']]

    if max_len != -1:
        if len(ret_url_list) > max_len: ret_url_list = ret_url_list[:max_len]

    return ret_url_list

# input : filepath
# output : url list
def getUrls_f(_filenames, url_type: int = 0, max_len:int = -1):
    ret_url_list = []
    if type(_filenames) == str:
            
        json_dict = loadJson(_filenames)

        ret_url_list = getUrls(json_dict, url_type, max_len)

    if type(_filenames) == list:
        for filename in _filenames:
            json_dict = loadJson(filename)

            ret_url_list.extend(getUrls(json_dict, url_type, max_len))

             
    return ret_url_list

# url_type = 0 일경우 benign 데이터, 1일 경우 phishing 데이터 가져옴
# 
def getFilePaths(dir_path, url_type:int = 0, **kwargs):

    shuffle = False
    max_len = -1
    path_list = []
    if 'shuffle' in kwargs.keys():
        shuffle = kwargs['shuffle']
    if 'max_len' in kwargs.keys():
        if kwargs['max_len'] <= 0: raise Exception("Invalid Parameter(max_len) Error!")
        max_len = kwargs['max_len']
    if url_type <0 or url_type >1: raise Exception("Invalid Parameter(url_type) Error!")

    #benign data
    if url_type == 0:
        if not os.path.isdir(dir_path):
            raise Exception(f"NOT directory Error! : {dir_path}")
        
        sub_dir_list = os.listdir(dir_path)
        sub_dir_list = [os.path.join(BENIGN_DIR, dir) for dir in sub_dir_list]
        sub_dir_list = [dir for dir in sub_dir_list if os.path.isdir(dir)]
        for dir in sub_dir_list:
            dir = os.path.join(dir, 'benign_data')
            sub_files_list = os.listdir(dir)
            sub_files_list = [os.path.join(dir, file) for file in sub_files_list]
            sub_files_list = [file for file in sub_files_list if os.path.isfile(file)]
            path_list.extend(sub_files_list)

    #phishing data
    if url_type == 1:
        if not os.path.isdir(dir_path):
            raise Exception(f"NOT directory Error! : {dir_path}")
        
        path_list = os.listdir(dir_path)
        path_list = [os.path.join(PHISHING_DIR, file) for file in path_list]
        path_list = [file for file in path_list if os.path.isfile(file)]

    # 만약 shuffle 옵션이 켜져있다면 랜덤으로 셔플한다.
    if shuffle: random.shuffle(path_list)

    # 만약 max_len 옵션이 켜져있다면 list길이를 조절한다.
    if max_len != -1:
        if len(path_list) > max_len: path_list = path_list[:max_len] 
    
    return path_list

def split_url(url):
    pass

if __name__ == '__main__':
    file_paths = getFilePaths(PHISHING_DIR, 1, shuffle=True)
    file_paths = file_paths[:30]

    print(getUrls_f(file_paths, 1, max_len=2))