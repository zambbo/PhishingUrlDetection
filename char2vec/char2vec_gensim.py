from this import d
from gensim.models import Word2Vec, KeyedVectors
from datasets import Char2VecDataset
import os
import sys
import torch.nn as nn
import torch
from config import *

def training():

    char2vec_dataset = Char2VecDataset(2, True, BENIGN_PATH, PHISHING_PATH)
    char_list  = char2vec_dataset.char_list
    print("training...")
    embedding_model = Word2Vec(char_list, sg=1, vector_size=100, window=5, workers=4)
    print("Finish training!")
    #key vector만 남긴다 (char와 그에 대응하는 임베딩 벡터)
    embedding_model.wv.save(MODEL_SAVE_PATH)

def testing():
    loaded_model = KeyedVectors.load(MODEL_SAVE_PATH)
    print(loaded_model.key_to_index)
    print(len(loaded_model.vectors))
    embedding = nn.Embedding.from_pretrained(torch.tensor(loaded_model.vectors, dtype=torch.float64))
    embedding.requires_grad = False
    print(loaded_model.key_to_index['2'])
    gensim_vector = torch.tensor(loaded_model['2'])
    embedding_vector = embedding(torch.tensor(loaded_model.key_to_index['2']))
    print(gensim_vector)
    print(embedding_vector)
    print(gensim_vector==embedding_vector)
    print(embedding)
    pass

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("TOO MANY ARGS!")
        sys.exit()

    if sys.argv[1] == 'train':
        training()
    elif sys.argv[1] == 'testing':
        testing()