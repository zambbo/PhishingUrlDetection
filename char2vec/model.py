import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Char2VecDatasetGENSIM
from config import *


def main():
    dataset = Char2VecDatasetGENSIM(False, BENIGN_PATH, PHISHING_PATH, char2vec_path = MODEL_SAVE_PATH, embedded_dim = 100)
    print("-"*20)
    print(dataset[0])


if __name__ == '__main__':
    main()