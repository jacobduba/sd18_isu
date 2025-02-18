from _typeshed import StrEnum
from pydantic import BaseModel

class Model(StrEnum):
    UNIXCODE_BASE = "microsoft/unixcode-base"

class Embedding(BaseModel):
    pass

class Vector(BaseModel):
    code_embedding: Embedding
    comment_embedding: Embedding
