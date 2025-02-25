import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

from datasets import load_dataset
from datasets.arrow_dataset import os
from models.code_search_net import DataPoint
from models.unix_coder import Pair
import torch
from torch import cuda, tensor
from torch import device as DeviceModel
from tqdm import tqdm

from unixcoder import UniXcoder

device = DeviceModel("cuda" if cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")

def generate_embedding(snippet_for_model: str, code_string: str) -> Pair:
    tokens_ids = model.tokenize(
        [snippet_for_model], max_length=512, mode="<encoder-only>"
    )
    source_ids = tensor(tokens_ids).to(device)
    _, nl_embedding = model(source_ids)

    return Pair(
        code_string=code_string,
        comment_embedding=torch.flatten(torch.nn.functional.normalize(nl_embedding, p=2, dim=1)).tolist(),
        comment_string=snippet_for_model,
    )


def create_code_search_net_dataset(slice_size:int = 20) -> List[DataPoint] | None:
    dataset = load_dataset(
        "code_search_net", "python", split="test", trust_remote_code=True
    )

    dataset = dataset.select(range(slice_size))

    data_points: List[DataPoint] = []
    for idx, item in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing dataset"
    ):
        try:
            data_point = DataPoint(
                id=idx,
                repository_name=item.get("repository_name", ""),
                func_path_in_repository=item.get("func_path_in_repository", ""),
                func_name=item.get("func_name", ""),
                whole_func_string=item.get("whole_func_string", ""),
                language=item.get("language", ""),
                func_code_string=item.get("func_code_string", ""),
                func_code_tokens=item.get("func_code_tokens", []),
                func_documentation_string=item.get("func_documentation_string", ""),
                func_documentation_string_tokens=item.get(
                    "func_documentation_string_tokens", []
                ),
                split_name="test",
                func_code_url=item.get("func_code_url", ""),
            )
            data_points.append(data_point)
        except Exception as e:
            print(f"Error processing item {idx} in test: {e}")

    return data_points


def process_data_point(dp: DataPoint) -> Pair:
    return generate_embedding(
        dp.func_documentation_string,
        dp.whole_func_string,
    )  # or change to generate_embedding_for_nl if needed


def process_data(data_points: List[DataPoint]) -> None:
    pairs: List[Pair] = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use executor.map to process data points
        for result in tqdm(
            executor.map(process_data_point, data_points),
            total=len(data_points),
            desc="Generating embeddings",
        ):
            pairs.append(result)

    # Writing all results to a file in one go
    try:
        with open("data.json", "w") as file:
            file.write("[\n")
            for idx, pair in enumerate(pairs):
                json.dump(pair.model_dump(), file, indent=4)
                if idx != len(pairs) - 1:
                    file.write(",")
                file.write("\n")
            file.write("\n]")
    except Exception as e:
        print(f"Error writing to file: {e}")
