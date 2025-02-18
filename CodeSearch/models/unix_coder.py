from enum import Enum
from typing import List, Union
from pydantic import BaseModel

# import unixcoder

class Model(str, Enum):
    UNIXCODE_BASE = "microsoft/unixcoder-base"

class Embedding(BaseModel):
    vector: Union[List[List[float]], List[float]]
    # model: Optional[unixcoder.UniXcoder] = None
    # length: Optional[int] = None

class Pair(BaseModel):
    code_embedding: Embedding
    comment_embedding: Embedding
