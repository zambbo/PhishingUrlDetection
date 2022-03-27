from model import PDTextCNN
from torch.utils.data import DataLoader, SubsetRandomSampler
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
from sklearn.model_selection import KFold
import numpy as np

def train(model, device, train_dataloader, optimizer):

    model.train()
    train_loss = 0.0
    true_pos, true_neg, false_pos, false_neg = 0.0, 0.0, 0.0, 0.0
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
        true_pos += torch.logical_and(label == 1, result == 1).sum().item()
        false_pos += torch.logical_and(label == 0, result == 1).sum().item()
        true_neg += torch.logical_and(label == 0, result == 0).sum().item()
        false_neg += torch.logical_and(label == 1, result == 0).sum().item()

    train_loss /= len(train_dataloader.dataset)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print(f"train\ttrue_pos: {true_pos}, true_neg: {true_neg}, false_pos: {false_pos}, false_neg: {false_neg}")
    try:
        precision = true_pos/ (true_pos + false_pos)
    except:
        precision = 0.00
    try:
        recall = true_pos / (true_pos + false_neg)
    except:
        recall = 0.00
    f1Score = 2*precision*recall/(precision + recall + 0.0000001)
    return train_loss, accuracy, precision, recall, f1Score


def evaluate(model, device, evaluate_dataloader):

    model.eval()

    test_loss = 0.0
    true_pos, true_neg, false_pos, false_neg = 0.0, 0.0, 0.0, 0.0
    for batch_id, (data, label) in tqdm(enumerate(evaluate_dataloader)):
        domain, path = data
        domain, path = domain.to(device), path.to(device)
        data = (domain, path)
        label = label.to(device)

        logit = model(data)
        loss = F.cross_entropy(logit, label)
        
        test_loss += loss.item()
        result = torch.max(logit, 1)[1]
        true_pos += torch.logical_and(label == 1, result == 1).sum().item()
        false_pos += torch.logical_and(label == 0, result == 1).sum().item()
        true_neg += torch.logical_and(label == 0, result == 0).sum().item()
        false_neg += torch.logical_and(label == 1, result == 0).sum().item()
    
    test_loss /= len(evaluate_dataloader.dataset)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print(f"test\ttrue_pos: {true_pos}, true_neg: {true_neg}, false_pos: {false_pos}, false_neg: {false_neg}")
    try:
        precision = true_pos/ (true_pos + false_pos)
    except:
        precision = 0.00
    try:
        recall = true_pos / (true_pos + false_neg)
    except:
        recall = 0.00
    f1Score = 2*precision*recall/(precision + recall + 0.0000001)
    return test_loss, accuracy, precision, recall, f1Score


def crossvalidation(domain_char2vec, path_char2vec, device, dataset, kfold_n, learning_rate, dim_channel, kernel_wins):

    kf = KFold(n_splits=kfold_n, shuffle=True)

    history = {'train_loss': [0]*EPOCHS,
    'val_loss': [0]*EPOCHS,
    'train_acc': [0]*EPOCHS,
    'val_acc': [0]*EPOCHS,
    'train_precision': [0]*EPOCHS,
    'val_precision': [0]*EPOCHS,
    'train_recall': [0]*EPOCHS,
    'val_recall': [0]*EPOCHS,
    'train_f1_score': [0]*EPOCHS,
    'val_f1_score': [0]*EPOCHS
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        model = PDTextCNN(domain_char2vec=domain_char2vec, path_char2vec=path_char2vec, embedding_dim=100, dim_channel=dim_channel, kernel_wins=kernel_wins, num_classes=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        for epoch in range(EPOCHS):
            print(f"fold: {fold} epoch: {epoch+1}")

            train_loss, train_accuracy, train_precision, train_recall, train_f1Score = train(model, device, train_loader, optimizer)
            val_loss, val_accuracy, val_precision, val_recall, val_f1Score = evaluate(model, device, val_loader)

            history['train_loss'][epoch] += train_loss
            history['train_acc'][epoch] += train_accuracy
            history['train_precision'][epoch] += train_precision
            history['train_recall'][epoch] += train_recall
            history['train_f1_score'][epoch] += train_f1Score
            history['val_loss'][epoch] += val_loss
            history['val_acc'][epoch] += val_accuracy
            history['val_precision'][epoch] += val_precision
            history['val_recall'][epoch] += val_recall
            history['val_f1_score'][epoch] += val_f1Score
        
        if fold == 0:
            avg_model_state_dict = model.state_dict()
        else:
            model_state_dict = model.state_dict()
            for key in model_state_dict:
                avg_model_state_dict[key] += model_state_dict[key]


    for epoch in range(EPOCHS):
        history['train_loss'][epoch] /= kfold_n
        history['train_acc'][epoch] /= kfold_n
        history['train_precision'][epoch] /= kfold_n
        history['train_recall'][epoch] /= kfold_n
        history['train_f1_score'][epoch] /= kfold_n
        history['val_loss'][epoch] /= kfold_n
        history['val_acc'][epoch] /= kfold_n
        history['val_precision'][epoch] /= kfold_n
        history['val_recall'][epoch] /= kfold_n
        history['val_f1_score'][epoch] /= kfold_n         

    for key in avg_model_state_dict:
        avg_model_state_dict[key] /= kfold_n
    model = PDTextCNN(domain_char2vec=domain_char2vec, path_char2vec=path_char2vec, embedding_dim=100, dim_channel=dim_channel, kernel_wins=kernel_wins, num_classes=2).to(device)
    model.load_state_dict(avg_model_state_dict)
    
    print(f"mean_loss: {history['val_loss'][-1]} mean_acc: {history['val_acc'][-1]}mean_precision: {history['val_precision'][-1]} mean_recall: {history['val_recall'][-1]} mean_f1_score: {history['val_f1_score'][-1]}")
    return history, model


def main():
    
    file_path = "./dataset/log_.txt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain_char2vec = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
    path_char2vec = KeyedVectors.load(CHAR2VEC_PATH_MODEL_SAVE_PATH)

    dataset = PDDataset(FINAL_DATASET_PATH, domain_max_len=100, path_max_len=100)

    test_len = floor(len(dataset) * TRAIN_TEST_SPLIT_RATIO)
    train_len = len(dataset) - test_len
    print(train_len, test_len)
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    learning_rate_candidate = [0.001, 0.005, 0.01]
    dim_channel_candidate = [5, 10, 15]
    kernel_wins_candidate = [[1,2,3], [2,3,4], [3,4,5]]

    test_loader = DataLoader(test_dataset, shuffle=True)

    best_test_f1Score = -1
    best_model_state_dict = None
    best_lr = -1
    best_dim_channel = -1
    best_kernel_wins = None
    for learning_rate in learning_rate_candidate:
        for dim_channel in dim_channel_candidate:
            for kernel_wins in kernel_wins_candidate:

                print('-'*10)
                print(f"learning_rate: {learning_rate}\tdim_channel: {dim_channel}\tkernel_wins: {kernel_wins}\n")
                print('-'*10+'\n')
                history, val_model = crossvalidation(domain_char2vec, path_char2vec, device, train_dataset, kfold_n=5, learning_rate=learning_rate, dim_channel=dim_channel, kernel_wins=kernel_wins)
                test_loss, test_accuracy, test_precision, test_recall, test_f1Score = evaluate(val_model, device, test_loader)
                print(history)
                print(f"test_loss: {test_loss}\ttest_acc: {test_accuracy}\ttest_precision: {test_precision}\ttest_recall: {test_recall}\ttest_f1_score: {test_f1Score}")
                if test_f1Score > best_test_f1Score:
                    best_test_f1Score = test_f1Score
                    best_model_state_dict = copy.deepcopy(val_model.state_dict())
                    best_lr = learning_rate
                    best_dim_channel = dim_channel
                    best_kernel_wins = kernel_wins
        
    
    print(f"best f1score: {best_test_f1Score}, best lr: {best_lr}, best_dim_channel: {best_dim_channel}, best_kernel_wins: {best_kernel_wins}")
    
    model = PDTextCNN(domain_char2vec=domain_char2vec, path_char2vec=path_char2vec, embedding_dim=100, dim_channel=best_dim_channel, kernel_wins=best_kernel_wins, num_classes=2).to(device) 
    model.load_state_dict(best_model_state_dict)
    torch.save(model, './dataset/model/textCNN_best_model.pt')


    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # model_best_state_dict = copy.deepcopy(model.state_dict())

    # for epoch in range(1, EPOCHS+1):

    #     tr_loss, tr_acc, tr_f1_score = train(model, device, train_loader, optimizer)
    #     print(f"Train Epoch : {epoch}\ttrain_loss : {tr_loss}\ttr_acc : {tr_acc}%\ttest_f1_score : {tr_f1_score}")

    #     val_loss, val_acc, val_f1_score = evaluate(model, device, test_loader)
    #     print(f"Test Epoch : {epoch}\ttest_loss : {val_loss}\ttest_acc : {val_acc}%\ttest_f1_score : {val_f1_score}")

    #     if val_acc > best_test_acc:
    #         best_test_acc = val_acc
    #         model_best_state_dict = copy.deepcopy(model.state_dict())
    #         print(f"model copy at accuracy {best_test_acc}")

    # model.load_state_dict(model_best_state_dict)

    #torch.save(model, BEST_MODEL_PATH)        
    




if __name__ == '__main__':
    main()