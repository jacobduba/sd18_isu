from models import unix_coder
from models.code_search_net import DataPoint
from models.unix_coder import Embedding
from unixcoder import UniXcoder
from torch import  cuda, tensor
from torch import device as DeviceModel
# import torch


device: DeviceModel = DeviceModel("cuda" if cuda.is_available() else "cpu")
model = UniXcoder(unix_coder.Model.UNIXCODE_BASE)  

def generate_embedding_for_func(dp: DataPoint) -> Embedding:
    return generate_embeddings(dp.whole_func_string)

def generate_embedding_for_nl(dp: DataPoint) -> Embedding:
    return generate_embeddings(dp.func_documentation_string)
    
def generate_embeddings(snippet: str) -> Embedding:
    tokens_ids = model.tokenize([snippet],max_length=512,mode="<encoder-only>")
    source_ids = tensor(tokens_ids).to(device)
    tokens_embeddings,nl_embedding = model(source_ids)

    return Embedding.model_validate([tokens_embeddings, nl_embedding])

