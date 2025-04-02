# Waits for some text input. Returns similiar code snippets from the training data.

import sqlite3
from typing import List, Tuple

import numpy as np
import torch
from unixcoder import UniXcoder

from data_processing import DB_FILE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")


def get_processed_data() -> List[Tuple[str, np.ndarray]]:
    """Loads code snippets and their embeddings from SQLite into memory."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()

            # Fetch all embeddings and corresponding code strings
            cursor.execute("SELECT code_string, embedding FROM embeddings")
            data = cursor.fetchall()

        # Convert BLOB (bytes) to NumPy arrays
        processed_data = [
            (row[0], np.frombuffer(row[1], dtype=np.float32)) for row in data
        ]

        return processed_data  # List of (code_string, embedding)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []


def process_user_code_segment(user_input: str) -> np.ndarray:
    # Encode user-given description
    tokens_ids = model.tokenize([user_input], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)

    # Get embeddings from the model
    _, nl_embedding = model(source_ids)

    # Detach, normalize, move to CPU, and flatten
    return (
        torch.flatten(torch.nn.functional.normalize(nl_embedding.detach(), p=2, dim=1))
        .cpu()
        .numpy()
    )


def get_top_ten(
    query_embedding: np.ndarray, processed_data: List[Tuple[str, np.ndarray]]
) -> List[Tuple[int, float]]:
    """Finds the top 10 most relevant code snippets using vector similarity."""
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Stack all embeddings into a matrix
    all_embeddings = np.stack([entry[1] for entry in processed_data])

    # Compute cosine similarity (dot product)
    scores = np.dot(all_embeddings, query_embedding)

    # Get top 10 indices sorted by highest similarity
    top_indices = np.argsort(scores)[::-1][:10]
    print("TOP_INDICES", top_indices)

    # Return the top 10 (index, score) pairs
    return [(idx, scores[idx]) for idx in top_indices]
