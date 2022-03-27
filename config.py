import os



BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_DIR = './dataset/model'
MODEL_NAME = 'textcnnNchar2vec_test.pt'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

TRAIN_TEST_SPLIT_RATIO = 0.2

NUM_WORKERS=4

DATASET_DIR = './dataset'
BENIGN_DIR = os.path.join(DATASET_DIR, 'benign_data')
PHISHING_DIR = os.path.join(DATASET_DIR, 'phishing_data')
BENIGN_PATH = os.path.join(DATASET_DIR, 'benigns.csv')
PHISHING_PATH = os.path.join(DATASET_DIR, 'phishings.csv')
FINAL_DATASET_PATH = os.path.join(DATASET_DIR, 'final_dataset.csv')

CHAR2VEC_DOMAIN_MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'model/domain_char2vec_final.cv')
CHAR2VEC_PATH_MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'model/path_char2vec_final.cv')