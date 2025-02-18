from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from datasets import load_dataset
from datasets.arrow_dataset import json, os
from models import unix_coder
from models.code_search_net import DataPoint
from models.unix_coder import Embedding, Pair
from torch import cuda, tensor
from torch import device as DeviceModel
from tqdm import tqdm
from unixcoder import UniXcoder

device = DeviceModel("cuda" if cuda.is_available() else "cpu")
model = UniXcoder(unix_coder.Model.UNIXCODE_BASE.value)


def generate_embedding_for_func(dp: DataPoint) -> Pair:
    return generate_embeddings(dp.whole_func_string)


def generate_embedding_for_nl(dp: DataPoint) -> Pair:
    return generate_embeddings(dp.func_documentation_string)


def generate_embeddings(snippet: str) -> Pair:
    tokens_ids = model.tokenize([snippet], max_length=512, mode="<encoder-only>")
    source_ids = tensor(tokens_ids).to(device)
    tokens_embeddings, _ = model(source_ids)

    return Pair(
        code_string=snippet,
        embedding=Embedding(vector=tokens_embeddings.squeeze().tolist()),
    )


def create_code_search_net_dataset() -> List[DataPoint] | None:
    # Load the dataset
    dataset = load_dataset(
        "code_search_net", "python", split="test", trust_remote_code=True
    )

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
                func_documentation_string_tokens=item.get("func_documentation_string_tokens", []),
                split_name="test",
                func_code_url=item.get("func_code_url", ""),
            )
            data_points.append(data_point)
        except Exception as e:
            print(f"Error processing item {idx} in test: {e}")

    return data_points


# Move the process_data_point function outside of process_data to make it callable by threads
def process_data_point(dp: DataPoint) -> Pair:
    return generate_embedding_for_func(dp)


def process_data(data_points: List[DataPoint]) -> None:
    """Processes the data points and writes embeddings to a file."""
    pairs: List[Pair] = []

    # Progress bar for generating embeddings
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_dp = {
            executor.submit(process_data_point, dp): dp for dp in data_points
        }

        with tqdm(
            as_completed(future_to_dp),
            total=len(data_points),
            desc="Generating embeddings",
        ) as gen_bar:
            for future in gen_bar:
                dp = future_to_dp[future]
                try:
                    result = future.result()
                    pairs.append(result)
                except Exception as e:
                    print(f"Error generating embedding for DataPoint ID {dp.id}: {e}")

    # Progress bar for writing to file
    with tqdm(total=len(pairs), desc="Writing embeddings to file") as write_bar:
        try:
            with open("data.json", "w") as file:
                for pair in pairs:
                    json.dump(pair.model_dump_json(), file, indent=4)
                    write_bar.update(1)
        except Exception as e:
            print(f"Error writing to file: {e}")
