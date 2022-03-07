import json

def getUrls(json_dict, url_type:int = 0):

    if url_type <0 or url_type >1: raise Exception("Invalid Parameter(url_type) Error!")

    if url_type == 0:
        ret_url_list = [obj['url'] for obj in json_dict]

    if url_type == 1:
        ret_url_list = [obj['url'] for obj in json_dict['_embedded']['phish']]


    return ret_url_list


