from this import d
from gensim.models import Word2Vec, KeyedVectors
#from datasets import Char2VecDataset
import os
import sys
sys.path.append('..')
import torch.nn as nn
import torch
from config import *
from utils import *
from tqdm import tqdm

# 미리 char2vec 모델을 훈련시키는 과정
def training_char2vec():
    # benign 300 * 10000개 phishing 15000 * 200개 정도의 json 파일을 훈련시키면 대충 50:50 정도의 비율로 훈련가능
    benign_file_paths = getFilePaths(BENIGN_DIR, 0, shuffle=True)
    phishing_file_paths = getFilePaths(PHISHING_DIR, 1, shuffle=True)

    for i in tqdm(range(300)):
        # benign url을 이용해 먼저 훈련 
        if i != 0:
            domain_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"domain_char2vec_{i}.cv"))
            path_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"path_char2vec_{i}.cv"))
            print(f"save{i}!")

        urls = getUrls_f(benign_file_paths[i], 0, 10000)
        try:
            _, domains, paths = split_urls(urls)
        except Exception as e:
            print(e)
            continue
        domains = list(map(url2charlist, domains))
        paths = list(map(url2charlist, paths))        
        if i==0:
            print(f'training-i!{i}')
            domain_embedding_model = Word2Vec(domains, vector_size=100, sg=1, window=5, min_count=1, workers=4, epochs=3)
            path_embedding_model = Word2Vec(paths, vector_size=100, sg=1, window=5, min_count=1, workers=4, epochs=3)
            print(f"finish training-i{i}")
        else:
            domain_embedding_model.build_vocab(domains, update=True)
            path_embedding_model.build_vocab(paths, update=True)
            domain_embedding_model.train(domains, total_examples=domain_embedding_model.corpus_count, epochs=3)
            path_embedding_model.train(paths, total_examples=path_embedding_model.corpus_count, epochs=3)

        # phishing url을 이용해서 훈련    
        for j in tqdm(range(i*50, (i+1)*50)):
            urls = getUrls_f(phishing_file_paths[j], 1, 200)
            try:
                _, domains, paths = split_urls(urls)
            except Exception as e:
                print(e)
                continue
            domains = list(map(url2charlist, domains))
            paths = list(map(url2charlist, paths))    
            if j==0:
                print(f'training-j!{j}')
                domain_embedding_model = Word2Vec(domains, vector_size=100, sg=1, window=5, min_count=1, workers=4, epochs=3)
                path_embedding_model = Word2Vec(paths, vector_size=100, sg=1, window=5, min_count=1, workers=4, epochs=3)
                print(f"finish training-j{j}")
            else:
                domain_embedding_model.build_vocab(domains, update=True)
                path_embedding_model.build_vocab(paths, update=True)
                domain_embedding_model.train(domains, total_examples=domain_embedding_model.corpus_count, epochs=3)
                path_embedding_model.train(paths, total_examples=path_embedding_model.corpus_count, epochs=3)                    
    domain_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"domain_char2vec_final.cv"))
    path_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"path_char2vec_final.cv"))


    # domain_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"domain_char2vec.cv"))
    # path_embedding_model.wv.save(os.path.join(MODEL_SAVE_DIR,f"path_char2vec.cv"))

    # char2vec_dataset = Char2VecDataset(2, True, BENIGN_PATH, PHISHING_PATH)
    # char_list  = char2vec_dataset.char_list
    # print("training...")
    # embedding_model = Word2Vec(char_list, sg=1, vector_size=100, window=5, workers=4)
    # print("Finish training!")
    # #key vector만 남긴다 (char와 그에 대응하는 임베딩 벡터)
    # embedding_model.wv.save(MODEL_SAVE_PATH)

def testing():
    loaded_model = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
    print(loaded_model.vectors[0])
    loaded_model.key_to_index['Unk'] = len(loaded_model.key_to_index)
    print(loaded_model.most_similar('ç'))
    # loaded_model = KeyedVectors.load(MODEL_SAVE_PATH)
    # print(loaded_model.key_to_index)
    # print(len(loaded_model.vectors))
    # embedding = nn.Embedding.from_pretrained(torch.tensor(loaded_model.vectors, dtype=torch.float64))
    # embedding.requires_grad = False
    # print(loaded_model.key_to_index['2'])
    # gensim_vector = torch.tensor(loaded_model['2'])
    # embedding_vector = embedding(torch.tensor(loaded_model.key_to_index['2']))
    # print(gensim_vector)
    # print(embedding_vector)
    # print(gensim_vector==embedding_vector)
    # print(embedding)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("TOO MANY ARGS!")
        sys.exit()

    if sys.argv[1] == 'train':
        training_char2vec()
    elif sys.argv[1] == 'testing':
        testing()