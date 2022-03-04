from this import d
from gensim.models import Word2Vec, KeyedVectors
from datasets import Char2VecDataset
import os
import sys

CUR_TO_DATASET_PATH = '../dataset/'
BENIGN_PATH = os.path.join(CUR_TO_DATASET_PATH, 'benigns_10000.csv')
PHISHING_PATH = os.path.join(CUR_TO_DATASET_PATH, 'phishings_10000.csv')
MODEL_SAVE_PATH = os.path.join(CUR_TO_DATASET_PATH, 'model/char2vec.cv')

def training():

    char2vec_dataset = Char2VecDataset(2, True, BENIGN_PATH, PHISHING_PATH)
    char_list  = char2vec_dataset.char_list
    print("training...")
    embedding_model = Word2Vec(char_list, sg=1, vector_size=100, window=5, workers=4)
    print("Finish training!")
    #key vector만 남긴다 (char와 그에 대응하는 임베딩 벡터)
    embedding_model.wv.save_word2vec_format(MODEL_SAVE_PATH)

def testing():
    loaded_model = KeyedVectors.load(MODEL_SAVE_PATH)
    print(loaded_model['h'])

    pass
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("TOO MANY ARGS!")
        sys.exit()

    if sys.argv[1] == 'train':
        training()
    elif sys.argv[1] == 'testing':
        testing()