import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch import cuda, tensor
from torch import device as DeviceModel
from tqdm import tqdm
from unixcoder import UniXcoder

from models.code_search_net import DataPoint

# SQLite database file
DB_FILE = "embeddings.db"
HAS_GPU = False

if torch.backends.mps.is_available():
    device = DeviceModel("mps")
    HAS_GPU = True
elif cuda.is_available():
    device = DeviceModel("cuda")
    HAS_GPU = True
else:
    device = DeviceModel("cpu")

model = UniXcoder("microsoft/unixcoder-base")
model.to(device)


def create_table(cursor: sqlite3.Cursor):
    """Creates the embeddings table if it doesn't exist."""
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            code_string TEXT,
            embedding BLOB
        )"""
    )


def embedding_exists(cursor: sqlite3.Cursor, index: int) -> bool:
    """Checks if an embedding already exists in the database."""
    cursor.execute("SELECT 1 FROM embeddings WHERE id = ?", (index,))
    return cursor.fetchone() is not None


def store_embedding_bulk(entries: List[Tuple[int, str, np.ndarray]], cursor: sqlite3.Cursor):
    cursor.executemany(
        "INSERT INTO embeddings (id, code_string, embedding) VALUES (?, ?, ?)",
        [(i, code, emb.tobytes()) for i, code, emb in entries],
    )


def store_embedding(
    index: int, code_string: str, embedding: np.ndarray, cursor: sqlite3.Cursor
):
    """Stores an embedding in the SQLite database."""
    cursor.execute(
        "INSERT INTO embeddings (id, code_string, embedding) VALUES (?, ?, ?)",
        (index, code_string, embedding.tobytes()),
    )


def generate_embeddings_ACCELERATED(
    data_points: List[DataPoint], cursor: sqlite3.Cursor
) -> List[Tuple[int, str, np.ndarray]] | None:
    filtered_data_points = filter(lambda dp: not embedding_exists(cursor, dp.id), data_points)
    tokens_ids = model.tokenize(
        [dp.func_documentation_string for dp in filtered_data_points],
        max_length=512,
        mode="<encoder-only>",
        padding=True,
    )
    source_ids = tensor(tokens_ids).to(device)
    with torch.no_grad():
        _, nl_embedding = model(source_ids)

    embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    entries = []
    for i, dp in enumerate(filtered_data_points):
        emb = torch.flatten(embedding[i]).detach().cpu().numpy()
        entries.extend([dp.id, dp.whole_func_string, emb])
    return entries


def generate_embedding(
    snippet_for_model: str, code_string: str, index: int, cursor: sqlite3.Cursor
):
    """Generates and stores an embedding if it doesn't already exist."""
    if embedding_exists(cursor, index):
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

    store_embedding(index, code_string, embedding, cursor)


def process_data(data_points: List[DataPoint], cursor: sqlite3.Cursor) -> None:
    """Processes data points in parallel and stores embeddings in SQLite."""
    if HAS_GPU:
        batch_size = 16
        with tqdm(
            total=len(data_points), desc="Generating & Storing embeddings (GPU)"
        ) as pbar:
            entries: List[Tuple[int, str, np.ndarray]] = []
            for i in range(0, len(data_points), batch_size):
                batch = data_points[i : i + batch_size]
                entries.extend(generate_embeddings_ACCELERATED(batch, cursor) or [])
                pbar.update(len(batch))
            store_embedding_bulk(entries, cursor)
    else:
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
                    desc="Generating & Storing embeddings (CPU)",
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


def create_code_search_net_dataset(slice_size: Optional[int] = None) -> List[DataPoint]:
    """Loads a subset of the CodeSearchNet dataset and returns it as DataPoint objects."""
    dataset: Any = load_dataset(
        "code_search_net",
        "python",
        split="train+test+validation",
        trust_remote_code=True,
    )
    if slice_size:
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
