from locale import MON_3
import os

CUR_TO_DATASET_PATH = '../dataset/'
BENIGN_PATH = os.path.join(CUR_TO_DATASET_PATH, 'benigns.csv')
PHISHING_PATH = os.path.join(CUR_TO_DATASET_PATH, 'phishings.csv')
MODEL_SAVE_DIR = os.path.join(CUR_TO_DATASET_PATH, 'model')
MODEL_SAVE_PATH = os.path.join(CUR_TO_DATASET_PATH, 'model/char2vec.cv')

CHAR2VEC_DOMAIN_MODEL_SAVE_PATH = os.path.join(CUR_TO_DATASET_PATH, 'model/domain_char2vec_final.cv')
CHAR2VEC_PATH_MODEL_SAVE_PATH = os.path.join(CUR_TO_DATASET_PATH, 'model/path_char2vec_final.cv')

BENIGN_DIR = os.path.join(CUR_TO_DATASET_PATH, 'benign_data')
PHISHING_DIR = os.path.join(CUR_TO_DATASET_PATH, 'phishing_data')

