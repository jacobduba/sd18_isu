from typing import List
from pydantic import BaseModel

class DataPoint(BaseModel):
    id: int
    repository_name: str
    func_path_in_repository: str
    func_name: str
    whole_func_string: str
    language: str
    func_code_string: str
    func_code_tokens: List[str]
    func_documentation_string: str
    func_documentation_string_tokens: List[str]
    split_name: str
    func_code_url: str



