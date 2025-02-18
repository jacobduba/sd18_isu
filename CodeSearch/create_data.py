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
#print(dataset)

#for what were doing I believe we only care about the train data
train_data = dataset["train"]

#This is just an example of how to use the dataset
num_data = int(input("Please input number of data pieces to print: "))
com_or_code = input("Would you like to print Code(1), Comments(2), else will print both: ")

for i in range(num_data):
    #this will give you the code of a piece of data
    data_code = train_data[i]["func_code_string"]
    data_comment = train_data[i]["func_documentation_string"]

    if(com_or_code == "1"):
        print(f"Function:{i+1}:\n Code:\n{data_code}\n{'-'*80}\n")
    #this will give you the comment of a piece of data, note its truncated to only 1 paragraph, dataset itself does this
    elif(com_or_code == "2"):
        print(f"Function:{i+1}:\n Comment:\n{data_comment}\n{'-'*80}\n")
    else:
        print(f"Function {i+1}:\n Comment:\n{data_comment}\n Code:\n{data_code}\n{'-'*80}\n")

