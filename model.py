import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *
from char2vec.datasets import Char2VecDatasetGENSIM, PDDataset
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

class PDTextCNN(nn.Module):

    def __init__(self, domain_char2vec, path_char2vec, embedding_dim, dim_channel, kernel_wins, num_classes):
        super(PDTextCNN, self).__init__()
        # 전체 vocab_size
        domain_vocab_size = len(domain_char2vec.key_to_index)
        path_vocab_size = len(path_char2vec.key_to_index)
        # 미리 훈련된 char2vec으로부터 임베딩 벡터 구성
        self.domain_embed = nn.Embedding.from_pretrained(torch.tensor(domain_char2vec.vectors,dtype=torch.float32))
        self.path_embed = nn.Embedding.from_pretrained(torch.tensor(path_char2vec.vectors, dtype=torch.float32))
        # 마지막 row에 "OOV" 를 위한 임베딩 벡터 추가로 생성

        domain_added_tensor = torch.tensor(np.random.rand(embedding_dim), dtype=torch.float32)
        domain_added_tensor = domain_added_tensor.unsqueeze(0)
        domain_concated_tensor = torch.cat([self.domain_embed.weight, domain_added_tensor],dim=0)

        path_added_tensor = torch.tensor(np.random.rand(embedding_dim), dtype=torch.float32)
        path_added_tensor = path_added_tensor.unsqueeze(0)
        path_concated_tensor = torch.cat([self.path_embed.weight, path_added_tensor], dim=0)

        self.domain_embed.weight = nn.Parameter(domain_concated_tensor)
        self.path_embed.weight = nn.Parameter(path_concated_tensor)

        # 이미 char2vec으로 훈련된 벡터값을 변경하지 않는다.
        self.domain_embed.requires_grad = False
        self.path_embed.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, embedding_dim)) for w in kernel_wins])

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(len(kernel_wins)*dim_channel, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)


    def tofc1(self, x, x_type=0):
        if x_type == 0:
            emb_x = self.domain_embed(x)
        elif x_type == 1:
            emb_x = self.path_embed(x)
        
        #2차원 데이터 (num of char, embed_dim) 인 텍스트 데이터를 3차원으로 확장하기 위해서 한차원 추가
        emb_x = emb_x.unsqueeze(1)

        con_x = [self.relu(conv(emb_x)) for conv in self.convs] 

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout1(fc_x)  

        fc_x = self.fc1(fc_x)
        return fc_x

    def forward(self, x):
        domain, path = x

        domain_fc_x = self.tofc1(domain, 0)
        path_fc_x = self.tofc1(path, 1)

        fc_x = torch.cat([domain_fc_x, path_fc_x], dim=1)
        
        fc_x = self.dropout2(fc_x)
        logit = self.fc2(fc_x)
        
        return logit

def main():

    dataset = PDDataset(benign_file_num=1, phishing_file_num=5, domain_max_len=100, path_max_len=100)
    print("-"*50)
    domain_char2vec = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
    path_char2vec = KeyedVectors.load(CHAR2VEC_PATH_MODEL_SAVE_PATH)

    model = PDTextCNN(domain_char2vec=domain_char2vec, path_char2vec=path_char2vec, embedding_dim=100, dim_channel=10, kernel_wins=[3,4,5], num_classes=2)

    for data, label in dataset:
        print(model(data))
        
    # dataset = Char2VecDatasetGENSIM(False, BENIGN_PATH, PHISHING_PATH, char2vec_path = MODEL_SAVE_PATH, embedded_dim = 100, max_length = 80)
    # print("-"*20)
    # print(dataset[0])
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # char2vec = KeyedVectors.load(MODEL_SAVE_PATH)
    # model = PDTextCNN(char2vec, 100, 10, [3,4,5], 2)
    # print(len(dataloader.dataset))
    # for batch_idx, (data, label) in enumerate(dataloader):
    #     print(f"batch idx : {batch_idx}")
    #     print(f"data : {data}")
    #     print(f"label : {label}")
    #     print(model(data))


if __name__ == '__main__':
    main()