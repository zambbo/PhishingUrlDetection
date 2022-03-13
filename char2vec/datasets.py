from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from config import *
import sys
sys.path.append('..')
from utils import *
from tqdm import tqdm
from sklearn.utils import shuffle
# 큰 csv파일을 읽어와서 chunksize별로 해보자
class PDDataset(Dataset):
    def __init__(self, benign_path, phishing_path, benign_url_num = -1, phishing_url_num = -1, domain_max_len = 100, path_max_len = 100):
        
        benign_chunk = pd.read_csv(benign_path, chunksize=1000)
        benign_chunk = list(benign_chunk)
        benign_df = pd.concat(benign_chunk)

        phishing_chunk = pd.read_csv(phishing_path, chunksize=1000)
        phishing_chunk = list(phishing_chunk)
        phishing_df = pd.concat(phishing_chunk)

        if benign_url_num != -1:
            benign_df = benign_df.iloc[:benign_url_num,:]
        if phishing_url_num != -1:
            phishing_df = phishing_df.iloc[:phishing_url_num,:]
        
        self.benign_df = shuffle(benign_df)
        self.phishing_df = shuffle(phishing_df)

        self.df = pd.concat([benign_df, phishing_df], axis=0)

        # domain과 path를 따로
        self.domain_char2vec = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
        self.path_char2vec = KeyedVectors.load(CHAR2VEC_PATH_MODEL_SAVE_PATH)

        self.domain_max_len = domain_max_len
        self.path_max_len = path_max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        url = self.df['url'].iloc[index]
        if type(url) != str:
            url = str(url)
        try:
            _, domain, path = split_url(url)
        except Exception as e:
            print(e)
            domain = url
            path = url

        label = self.df['label'].iloc[index]

        encoded_domain = self.encoding(domain, 0)
        encoded_path = self.encoding(path, 1)

        encoded_domain = torch.tensor(encoded_domain, dtype=torch.long)
        encoded_path = torch.tensor(encoded_path, dtype=torch.long)

        label = torch.tensor(label, dtype=torch.long)


        return (encoded_domain, encoded_path), label

    def encoding(self, url, domainOrPath = 0):
        
        # string -> char list
        char_list = url2charlist(url)

        #domain
        if domainOrPath == 0:
            char2vec = self.domain_char2vec
            max_len = self.domain_max_len
        #path
        elif domainOrPath == 1:
            char2vec = self.path_char2vec
            max_len = self.path_max_len
        
        encoded_vector = np.full(max_len, len(char2vec.key_to_index)) # OOV단어의 경우에는 단어장 제일 끝 번호로 초기화

        
        for i in range(min(len(char_list), max_len)):
            
            if char_list[i] in char2vec.key_to_index.keys():
                encoded_vector[i] = char2vec.key_to_index[char_list[i]]
            else:
                encoded_vector[i] = len(char2vec.key_to_index)
        
        return encoded_vector

# class PDDataset(Dataset):
#     def __init__(self, benign_file_num:int = 5, phishing_file_num:int = 300, domain_max_len = 100, path_max_len = 100):

#         self.benign_file_num = benign_file_num
#         self.phishing_file_num = phishing_file_num

#         self.all_url_num = 0

#         benign_file_paths = getFilePaths(BENIGN_DIR, 0, shuffle=True)
#         phishing_file_paths = getFilePaths(PHISHING_DIR, 1, shuffle=True)

#         self.benign_file_paths = benign_file_paths[:benign_file_num]
#         self.phishing_file_paths = phishing_file_paths[:phishing_file_num]


#         # domain과 path를 따로
#         self.domain_char2vec = KeyedVectors.load(CHAR2VEC_DOMAIN_MODEL_SAVE_PATH)
#         self.path_char2vec = KeyedVectors.load(CHAR2VEC_PATH_MODEL_SAVE_PATH)

#         self.domain_max_len = domain_max_len
#         self.path_max_len = path_max_len
    
#     def __len__(self):
#         return self.benign_file_num + self.phishing_file_num
    
#     def __getitem__(self, index):

#         if index < self.benign_file_num:
#             url_type = 0 # benign
#             file_paths = self.benign_file_paths
#         else:
#             url_type = 1 # phishing
#             file_paths = self.phishing_file_paths
#             index -= self.benign_file_num

#         datas, label = getDataSetNLabel(file_paths[index], url_type=url_type)

#         encoded_domains = []
#         encoded_paths = []
#         for data in tqdm(datas):
#             _, domain, path = data

#             encoded_domains.append(self.encoding(domain, 0))
#             encoded_paths.append(self.encoding(path, 1))

#         self.all_url_num += len(encoded_domains)

#         encoded_domains = np.stack(encoded_domains, axis=0)
#         encoded_paths = np.stack(encoded_paths, axis=0)

#         encoded_domains = torch.tensor(encoded_domains, dtype=torch.long)
#         encoded_paths = torch.tensor(encoded_paths, dtype=torch.long)

#         label = torch.tensor(label, dtype=torch.long)


#         return (encoded_domains, encoded_paths), label

#     def encoding(self, url, domainOrPath = 0):
        
#         # string -> char list
#         char_list = url2charlist(url)

#         #domain
#         if domainOrPath == 0:
#             char2vec = self.domain_char2vec
#             max_len = self.domain_max_len
#         #path
#         elif domainOrPath == 1:
#             char2vec = self.path_char2vec
#             max_len = self.path_max_len
        
#         encoded_vector = np.full(max_len, len(char2vec.key_to_index)) # OOV단어의 경우에는 단어장 제일 끝 번호로 초기화

        
#         for i in range(min(len(char_list), max_len)):
            
#             if char_list[i] in char2vec.key_to_index.keys():
#                 encoded_vector[i] = char2vec.key_to_index[char_list[i]]
#             else:
#                 encoded_vector[i] = len(char2vec.key_to_index)
        
#         return encoded_vector
            
        

#word2vec을 이용한 dataset
class Char2VecDatasetGENSIM(Dataset):
    def __init__(self, training=False, json:bool = True, *file_paths, **kwargs):
            
        # path가 1개 이상일 경우 concatenate해서 df를 구성한다.
        if len(file_paths) > 1:
            print("Reading File...")
            df_list = [pd.read_csv(file_path, index_col=0) for file_path in file_paths]
            df = pd.concat(df_list, axis=0, ignore_index=True)
            print("Finish Reading!")
        elif len(file_paths) == 1:
            print("Reading File...")
            df = pd.read_csv(file_paths, index_col=0)
            print("Finish Reading!")
        else:
            raise Exception("No File Path Exception")
        
        # 행별로 url을 character별로 쪼개서 리스트로 가공한다.
        df['url'] = df.apply(lambda x: [char for char in x['url']], axis=1)

        if len(kwargs) > 1:
            for key, item in kwargs.items():
                # 만약 char2vec 모델의 path를 인자로 받았으면 불러온다.
                if key=='char2vec_path': 
                    self.char2vec_model = KeyedVectors.load(item)
                if key=='embedded_dim':
                    self.embedded_dim = item
                if key=='max_length':
                    self.max_length = item
        elif len(kwargs) == 1:
            if 'char2vec_path' in kwargs.keys(): self.char2vec_model = KeyedVectors.load(kwargs['char2vec_path'])

        #만약 training == True 면 word2vec을 훈련시켜 embedding으로 사용한다.
        if training:
            print("training!")
            char2vec_model = Word2Vec(df['url'], vector_size=self.embedded_dim, window=5, sg=1, workers=4)
            self.char2vec_model = char2vec_model.wv
            print("finish")
        # chars를 embedded vector로 변환하여 저장


        def row_func(x):
            return_arr = np.full(80,len(self.char2vec_model.key_to_index),dtype=np.int64)

            # 최대 max_length까지 맞춘다.
            if len(x) > self.max_length:
                x = x[:self.max_length]

            for i in range(len(x)):
                if x[i] in self.char2vec_model.key_to_index.keys():
                    return_arr[i] = self.char2vec_model.key_to_index[x[i]]
                else:
                    return_arr[i] = len(self.char2vec_model.key_to_index)
            return return_arr
        embedded_chars = df['url'].apply(row_func).values
        
        """
            url을 char단위로 정수 인코딩하여 저장
            self.data = [
                [0, 1, 2, 3, -1],
                [1, 6, 3, -1, -1],
                ...
                [2, 3, 10, 23, 3]
            ]
        """
        embedded_chars = np.stack(embedded_chars, axis=0)
        self.data = torch.tensor(embedded_chars, dtype=torch.long)
        self.label = torch.tensor(df['label'].values, dtype=torch.long)
        

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)      


# 직접 word2vec을 훈련시킬때(미완성)
class Char2VecDataset(Dataset):

    def __init__(self, window_size=2, gensim=False, *file_paths):
        # path가 1개 이상일 경우 concatenate해서 df를 구성한다.
        if len(file_paths) > 1:
            print("Reading File...")
            df_list = [pd.read_csv(file_path, index_col=0) for file_path in file_paths]
            df = pd.concat(df_list, axis=0, ignore_index=True)
            print("Finish Reading!")
        elif len(file_paths) == 1:
            print("Reading File...")
            df = pd.read_csv(file_paths, index_col=0)
            print("Finish Reading!")
        else:
            raise Exception("No File Path Exception")

        self.window_size = window_size
        if gensim:
            url_list = df['url'].values
            print("making char_list...")
            print(url_list)
            char_list = [[*url] for url in url_list if type(url) == str or type(url) == np.str]
            print("finish making char_list...")
            self.char_list = char_list
        else:
            vocab, char_to_idx, idx_to_char, data = self.preprocessing(df)
            self.vocab = vocab
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
            self.data = torch.tensor(data, dtype=torch.long)

    # (center, context) 튜플 형태로 반환 center가 data context가 label역할을 함 center로부터 context를 맞추는
    def __getitem__(self, idx):
        center = self.data[idx, 0]
        context = self.data[idx, 1]
        return center, context
    
    def __len__(self):
        return len(self.data)
    
    def preprocessing(self, df):
        url_list = df['url'].values

        # 모든 url로부터 중복되지 않는 character를 뽑아내서 vocabulary로 구성함
        vocab = set([char for url in url_list for char in url])
        
        # character로부터 정수 인코딩 된 값을 얻는 dictionary 객체
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        # 정수 인코딩 값으로부터 character를 얻는 dictionary 객체
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}

        data = []
        # 모든 url에 대해서 반복한다
        for url in url_list:
            #url의 모든 character를 정수 인코딩 하여 배열로 저장한다.
            indices = [char_to_idx[char] for char in url]
            
            #url의 모든 인덱스를 돌아가면서 중심 character로 지정한다.
            for center_char_pos in range(len(indices)):
                
                # window size 만큼 양옆의 character를 context character로 구성한다.
                for w in range(-self.window_size, self.window_size + 1):
                    context_char_pos = center_char_pos + w

                    # 만약 범위를 벗어나거나 center_character_pos와 같은 인덱스일 경우 continue
                    if context_char_pos < 0 or context_char_pos >= len(indices) or context_char_pos == center_char_pos: continue

                    center_char_idx = indices[center_char_pos]
                    context_char_idx = indices[context_char_pos]

                    """
                    최종 데이터 형태는 
                    [
                        [center, context],
                        [center, context],
                        ...
                    ]
                    """
                    data.append([center_char_idx, context_char_idx])


        return vocab, char_to_idx, idx_to_char, data

if __name__ == '__main__':
    ds = PDDataset(BENIGN_PATH, PHISHING_PATH)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    for (domain, path), label in tqdm(dl):
        print(domain)
        print(path)
        print(label)
