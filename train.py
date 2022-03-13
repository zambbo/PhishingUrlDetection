from model import PDTextCNN
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from char2vec.datasets import Char2VecDatasetGENSIM, PDDataset
import torch.nn.functional as F
import torch.optim as optim
import torch
from config import *
from gensim.models import KeyedVectors
from math import floor
import copy
from tqdm import tqdm

def train(model, device, train_dataloader, optimizer):

    model.train()
    corrects, train_loss = 0.0, 0

    for batch_idx, (data, label) in tqdm(enumerate(train_dataloader)):
        
        # data와 label을 device에 fetch한다.
        domain, path = data
        domain, path = domain.to(device), path.to(device)
        data = (domain, path)
        label = label.to(device)
        
        # 이전 배치에서 계산된 optimizer의 gradient값을 초기화
        optimizer.zero_grad()

        logit = model(data) 


        loss = F.cross_entropy(logit, label)
        
        #역전파
        loss.backward()
        #파라미터 업데이트
        optimizer.step()

        train_loss += loss.item()
        result = torch.max(logit, dim=1)[1] # [1]의 의미는 index를 뽑아오는 거다. torch.max 함수는 값과 index를 튜플 형태로 반환
        corrects += result.eq(label).sum()

    train_loss /= len(train_dataloader.dataset)
    accuracy = 100.0 * corrects / len(train_dataloader.dataset)

    return train_loss, accuracy

def evaluate(model, device, evaluate_dataloader):

    model.eval()

    corrects, test_loss = 0.0, 0

    for batch_id, (data, label) in tqdm(enumerate(evaluate_dataloader)):
        domain, path = data
        domain, path = domain.to(device), path.to(device)
        data = (domain, path)
        label = label.to(device)

        logit = model(data)
        loss = F.cross_entropy(logit, label)
        
        test_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += result.eq(label).sum()
    
    test_loss /= len(evaluate_dataloader.dataset)
    accuracy = 100 * corrects / len(evaluate_dataloader.dataset)

    return test_loss, accuracy






def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain_char2vec = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
    path_char2vec = KeyedVectors.load(CHAR2VEC_PATH_MODEL_SAVE_PATH)

    model = PDTextCNN(domain_char2vec=domain_char2vec, path_char2vec=path_char2vec, embedding_dim=100, dim_channel=10, kernel_wins=[3,4,5], num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_test_acc = -1

    dataset = PDDataset(BENIGN_PATH, PHISHING_PATH)

    test_len = floor(len(dataset) * TRAIN_TEST_SPLIT_RATIO)
    train_len = len(dataset) - test_len
    print(train_len, test_len)
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model_best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS+1):

        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        print(f"Train Epoch : {epoch}\ttrain_loss : {tr_loss}\ttr_acc : {tr_acc}%")

        val_loss, val_acc = evaluate(model, device, test_loader)
        print(f"Test Epoch : {epoch}\ttest_loss : {val_loss}\ttest_acc : {val_acc}%")

        if val_acc > best_test_acc:
            best_test_acc = val_acc
            model_best_state_dict = copy.deepcopy(model.state_dict())
            print(f"model copy at accuracy {best_test_acc}")

    model.load_state_dict(model_best_state_dict)

    #torch.save(model, BEST_MODEL_PATH)        
    




if __name__ == '__main__':
    main()