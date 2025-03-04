from typing import List
from pydantic import BaseModel

class Pair(BaseModel):
    code_string: str
    comment_embedding: List[float]
    comment_string: str
