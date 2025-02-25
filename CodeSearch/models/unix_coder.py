from enum import Enum
from typing import List 
from pydantic import BaseModel


class Model(str, Enum):
    UNIXCODE_BASE = "microsoft/unixcoder-base"


class Embedding(BaseModel):
    vector: List[List[float]]


class Pair(BaseModel):
    code_string: str
    comment_embedding: Embedding
