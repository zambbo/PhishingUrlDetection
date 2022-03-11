import os



BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_DIR = './dataset/model'
MODEL_NAME = 'textcnnNchar2vec.pt'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

TRAIN_TEST_SPLIT_RATIO = 0.2

NUM_WORKERS=4

DATASET_DIR = './dataset'
BENIGN_DIR = os.path.join(DATASET_DIR, 'benign_data')
PHISHING_DIR = os.path.join(DATASET_DIR, 'phishing_data')