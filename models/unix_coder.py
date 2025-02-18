from _typeshed import StrEnum
from typing import List, Optional
from pydantic import BaseModel

import unixcoder

class Model(StrEnum):
    UNIXCODE_BASE = "microsoft/unixcode-base"

class Embedding(BaseModel):
    vector: List[float]
    model: Optional[unixcoder.UniXcoder] = None
    length: Optional[int] = None

class Pair(BaseModel):
    code_embedding: Embedding
    comment_embedding: Embedding
