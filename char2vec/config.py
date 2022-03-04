import os
import sys

CUR_TO_DATASET_PATH = '../dataset/'
BENIGN_PATH = os.path.join(CUR_TO_DATASET_PATH, 'benigns_10000.csv')
PHISHING_PATH = os.path.join(CUR_TO_DATASET_PATH, 'phishings_10000.csv')
MODEL_SAVE_PATH = os.path.join(CUR_TO_DATASET_PATH, 'model/char2vec.cv')