from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class Char2VecDataset(Dataset):

    def __init__(self, file_path, window_size=2):
        
        df = pd.read_csv(file_path, index_col = 0)
        self.window_size = window_size
        vocab, char_to_idx, idx_to_char, data = self.preprocessing(df)
        self.vocab = vocab
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.data = torch.tensor(data, dtype=torch.long)

    # (center, context) 튜플 형태로 반환 center가 data context가 label역할을 함 center로부터 context를 맞추는
    def __item__(self, idx):
        center = self.data[idx, 0]
        context = self.data[idx, 1]
        return center, context
    
    def __len__(self):
        return len(self.data)
    
    def preprocessing(self, df):
        url_list = df['url'].values
        
        # 모든 url로부터 중복되지 않는 character를 뽑아내서 vocabulary로 구성함
        vocab = set(np.array([[*url] for url in url_list]).flatten())
        
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