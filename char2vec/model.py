import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Char2Vec(nn.Module):

    def __init__(self, embedding_dim, vocab):
        
        self.embed1 = nn.Embedding(len(vocab), embedding_dim)
        
        pass


    def forward(self, x):