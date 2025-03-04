# Waits for some text input. Returns similiar code snippets from the training data.

import json
import numpy as np
import torch
from typing import List, Tuple
from models.unix_coder import Pair
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")


def get_processed_data() -> List[Pair]:
    try:
        with open('data.json', 'r') as file:
            # contains objects that have an array of numbers and a code string
            json_data = json.load(file)

            # get the code strings and embeddings from json.data
            validated_pairs = []
            for item in json_data:
                validated_pairs.append(Pair.model_validate(item))
            
            return validated_pairs
    except FileNotFoundError:
        print("The data.json file does not exist")
    except json.JSONDecodeError:
        print("Error decoding data.json")

    return []


def process_user_code_segment(user_input: str) -> List[float]:
    # encode user given description
    tokens_ids = model.tokenize([user_input],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    _, nl_embedding = model(source_ids)

    # Normalize embedding and flatten it
    return torch.flatten(torch.nn.functional.normalize(nl_embedding.clone().detach(), p=2, dim=1)).tolist()


def get_top_ten(processed_user_code: List[float], processed_data: List[Pair]) -> List[Tuple[int, float]]:
    code_segment_map = {}
    for index, pair in enumerate(processed_data):
        # calculate similarity with the dot product and store the value
        code_segment_map[index] = np.dot(pair.comment_embedding, processed_user_code)

    # get top 10 most similar code segments
    return sorted(code_segment_map.items(), key=lambda x: x[1], reverse=True)[:10]
