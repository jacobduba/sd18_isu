from enum import Enum
from typing import List, Union
from pydantic import BaseModel


class Model(str, Enum):
    UNIXCODE_BASE = "microsoft/unixcoder-base"


class Embedding(BaseModel):
    vector: Union[List[List[float]], List[float]]


class Pair(BaseModel):
    code_string: str
    embedding: Embedding
