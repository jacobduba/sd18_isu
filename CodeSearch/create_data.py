# This file created a .npy file containing embeddings for each code example in a dataset.

import torch
#new dependency with datasets for huggingface
#uv pip install datasets
import datasets
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

from datasets import load_dataset

# Load the dataset
#needed to add the trust setting as code_search required it, could be bad
#seems to be needed to structure the dataset?
#NOTICE THIS LOAD WILL DOWNLOAD THE DATASET LOCALLY!!! abt 3 gigs total
dataset = load_dataset("code-search-net/code_search_net", trust_remote_code=True)
#simply displays "datasplits" and each splits data fields 
print(dataset)
