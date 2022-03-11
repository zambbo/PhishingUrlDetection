import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from char2vec.config import *
from char2vec.datasets import Char2VecDatasetGENSIM
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

class PDTextCNN(nn.Module):

    def __init__(self, char2vec, embedding_dim, dim_channel, kernel_wins, num_classes):
        super(PDTextCNN, self).__init__()
        # 전체 vocab_size
        vocab_size = len(char2vec.key_to_index)
        # 미리 훈련된 char2vec으로부터 임베딩 벡터 구성
        self.embed = nn.Embedding.from_pretrained(torch.tensor(char2vec.vectors,dtype=torch.float32))
        # 마지막 row에 "OOV" 를 위한 임베딩 벡터 추가로 생성
        added_tensor = torch.tensor(np.random.rand(embedding_dim), dtype=torch.float32)
        added_tensor = added_tensor.unsqueeze(0)
        concated_tensor = torch.cat([self.embed.weight, added_tensor],dim=0)
        self.embed.weight = nn.Parameter(concated_tensor)

        # 이미 char2vec으로 훈련된 벡터값을 변경하지 않는다.
        self.embed.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, embedding_dim)) for w in kernel_wins])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_classes)
        
    def forward(self, x):
        
        emb_x = self.embed(x)
        
        #2차원 데이터 (num of char, embed_dim) 인 텍스트 데이터를 3차원으로 확장하기 위해서 한차원 추가
        emb_x = emb_x.unsqueeze(1)

        con_x = [self.relu(conv(emb_x)) for conv in self.convs] 

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)

        logit = self.fc(fc_x)

        return logit

def main():
    dataset = Char2VecDatasetGENSIM(False, BENIGN_PATH, PHISHING_PATH, char2vec_path = MODEL_SAVE_PATH, embedded_dim = 100, max_length = 80)
    print("-"*20)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    char2vec = KeyedVectors.load(MODEL_SAVE_PATH)
    model = PDTextCNN(char2vec, 100, 10, [3,4,5], 2)
    print(len(dataloader.dataset))
    # for batch_idx, (data, label) in enumerate(dataloader):
    #     print(f"batch idx : {batch_idx}")
    #     print(f"data : {data}")
    #     print(f"label : {label}")
    #     print(model(data))


if __name__ == '__main__':
    main()