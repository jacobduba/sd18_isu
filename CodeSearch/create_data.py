# This file created a .npy file containing embeddings for each code example in a dataset.

import torch
from datasets import load_dataset
from models.unix_coder import Embedding, Pair

from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)
