import sqlite3
from datasets import load_dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from models.code_search_net import DataPoint
import torch
from torch import cuda, tensor
from torch import device as DeviceModel
from tqdm import tqdm

from unixcoder import UniXcoder

# SQLite database file
DB_FILE = "embeddings.db"

# Initialize UniXcoder
device = DeviceModel("cuda" if cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")


def create_table():
    """Creates the embeddings table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                code_string TEXT,
                embedding BLOB
            )"""
        )
        conn.commit()


def embedding_exists(index: int) -> bool:
    """Checks if an embedding already exists in the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM embeddings WHERE id = ?", (index,))
        return cursor.fetchone() is not None


def store_embedding(index: int, code_string: str, embedding: np.ndarray):
    """Stores an embedding in the SQLite database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO embeddings (id, code_string, embedding) VALUES (?, ?, ?)",
            (index, code_string, embedding.tobytes()),
        )
        conn.commit()


def generate_embedding(snippet_for_model: str, code_string: str, index: int):
    """Generates and stores an embedding if it doesn't already exist."""
    if embedding_exists(index):
        return  # Skip processing if already exists

    tokens_ids = model.tokenize(
        [snippet_for_model], max_length=512, mode="<encoder-only>"
    )
    source_ids = tensor(tokens_ids).to(device)
    _, nl_embedding = model(source_ids)

    # Convert to NumPy array and normalize
    embedding = (
        torch.flatten(torch.nn.functional.normalize(nl_embedding, p=2, dim=1))
        .detach()  # Detach from computation graph
        .cpu()
        .numpy()
    )

    store_embedding(index, code_string, embedding)


def process_data(data_points: List[DataPoint]) -> None:
    """Processes data points in parallel and stores embeddings in SQLite."""
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda dp: generate_embedding(
                        dp.func_documentation_string, dp.whole_func_string, dp.id
                    ),
                    data_points,
                ),
                total=len(data_points),
                desc="Generating & Storing embeddings",
            )
        )


def load_embeddings():
    """Loads all embeddings from SQLite into memory for fast search."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM embeddings")
        rows = cursor.fetchall()

    ids, embeddings = [], []
    for idx, blob in rows:
        ids.append(idx)
        embeddings.append(np.frombuffer(blob, dtype=np.float32))

    return np.array(ids), np.stack(embeddings)


def search(query_embedding: np.ndarray, top_k: int = 10):
    """Performs a fast search using NumPy dot product."""
    ids, embeddings_matrix = load_embeddings()

    # Compute cosine similarity
    scores = np.dot(embeddings_matrix, query_embedding)

    # Get top K results
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(ids[i], scores[i]) for i in top_indices]


def create_code_search_net_dataset(slice_size: int = 2000) -> List[DataPoint]:
    """Loads a subset of the CodeSearchNet dataset and returns it as DataPoint objects."""
    dataset: Any = load_dataset(
        "code_search_net", "python", split="test", trust_remote_code=True
    )
    dataset = dataset.select(range(slice_size))

    data_points: List[DataPoint] = []
    for idx, item in tqdm(
        enumerate(dataset), total=len(dataset), desc="Loading dataset"
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
            print(f"Error processing item {idx}: {e}")

    return data_points
