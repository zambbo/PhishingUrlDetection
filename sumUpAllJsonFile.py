import pandas as pd
import os
import json
import sys


# get number of files upto N
N = 15000

def decoding_json(_filename):
    try:
        with open(_filename, "r") as json_file:
            return json.load(json_file)
    except:
        raise Exception(f'Error While Opening {_filename}...')

def joining_target_directory_path(_base_file_name, _target_directory):
    target_file_list = os.listdir(_target_directory)
    target_file_list = [os.path.join(_base_file_name, target_path) for target_path in target_file_list]
    return target_file_list

def saving_json(_target_file_name, _data):
    with open(_target_file_name, 'w') as json_file:
        json.dump(_data, json_file)

def phishing():
    base_file_name = './dataset/phishing_data'
    target_directory = './dataset/phishing_data'
    saving_file_name = './dataset/after_extract_phishing_url.json'

    target_file_list = joining_target_directory_path(base_file_name, target_directory)

    if len(target_file_list) > N: target_file_list = target_file_list[:N]
    phishing_url_list = []

    for idx, target_file in enumerate(target_file_list):
        try:
            target_json = decoding_json(target_file)
            phishing_urls = [obj['url'] for obj in target_json['_embedded']['phish']]
            phishing_url_list.extend(phishing_urls)
            print(f"finish {target_file}! {idx+1}/{len(target_file_list)}")
        except Exception as e:
            print(e)

    saving_json(saving_file_name, phishing_url_list)

def getAllBenignJsonFile(_base_url):
    all_file = os.listdir(_base_url)
    all_file = [os.path.join(_base_url, file_name) for file_name in all_file]
    target_dirs = [path for path in all_file if os.path.isdir(path)]
    
    target_dirs = [os.path.join(path,"benign_data") for path in target_dirs]

    import re

    matchPattern = re.compile('json$')

    target_files = []

    for dirs in target_dirs:
        files_under_dir = os.listdir(dirs)
        files_under_dir = [file_name for file_name in files_under_dir if matchPattern.search(file_name) is not None]
        files_under_dir = [os.path.join(dirs, file_name) for file_name in files_under_dir]
        target_files.extend(files_under_dir)
    return target_files

def benign():
    base_directory_name = './dataset/benign_data'
    saving_file_name = './dataset/after_extract_benign_url.json'
    
    target_file_list = getAllBenignJsonFile(base_directory_name)

    if len(target_file_list) > N: target_file_list = target_file_list[:N]
    benign_url_list = []

    for idx, target_file in enumerate(target_file_list):
        try:
            target_json = decoding_json(target_file)
            phishing_urls = [obj['url'] for obj in target_json]
            benign_url_list.extend(phishing_urls)
            print(f"finish {target_file}! {idx+1}/{len(target_file_list)}")
        except Exception as e:
            print(e)

    saving_json(saving_file_name, benign_url_list)

def main():
    if len(sys.argv) != 2:
        print("Too many arguments!")
        sys.exit()
    
    target_type = sys.argv[1]

    if target_type == 'benign':
        benign()
    elif target_type == 'phishing':
        phishing()
    else:
        print("Wront argument name!")
        sys.exit()

if __name__ == '__main__':
    main()