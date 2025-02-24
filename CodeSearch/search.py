# Waits for some text input. Returns similiar code snippets from the training data.

import json
import numpy as np
import torch
from typing import List, Tuple
from unixcoder import UniXcoder
from models.unix_coder import Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")


def get_processed_data() -> Tuple[List[str], np.ndarray]:
    try:
        with open('data.json', 'r') as file:
            # contains objects that have an array of numbers and a code string
            json_data = json.load(file)

            # get the code strings and embeddings from json.data
            return [item["code_string"] for item in json_data], np.array([item["embedding"]["vector"] for item in json_data])
    except FileNotFoundError:
        print("The data.json file does not exist")
    except json.JSONDecodeError:
        print("Error decoding data.json")

    return [], np.array([])


def process_user_code_segment(user_input: str) -> torch.Tensor:
    # encode user given description
    tokens_ids = model.tokenize([user_input],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, _ = model(source_ids)

    embedding=Embedding(vector=tokens_embeddings.squeeze().tolist())

    # Normalize embedding
    return torch.nn.functional.normalize(torch.tensor(embedding), p=2, dim=1)


def get_top_ten(processed_user_code: torch.Tensor, embedding_list: np.ndarray) -> list:
    code_segment_map = {}
    for index, values in enumerate(embedding_list):
        # calculate similarity with the dot product and store the value
        # code_segment_map[index] = np.dot(values, process_user_code_segment(user_input))

        # I'm not sure if dot product is desired, but this was given in example code as a similarity check function
        code_segment_map[index] = torch.einsum("ac,bc->ab", processed_user_code, torch.nn.functional.normalize(values, p=2, dim=1))

    # get top 10 most similar code segments
    return sorted(code_segment_map.items(), key=lambda x: x[1], reverse=True)[:10]


if __name__ == "__main__":
    user_input = input("Enter your code description: ")
    code_segment_list, embedding_list = get_processed_data()
    processed_user_code = process_user_code_segment(user_input)
    top_ten = get_top_ten(processed_user_code, embedding_list)
    counter = 1
    for index in top_ten:
        print(f"{counter}: {code_segment_list[index]} with a similarity score of {top_ten[index]}")
        counter += 1
