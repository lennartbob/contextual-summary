


from enum import StrEnum
from attrs import define

class ReasoningType(StrEnum):
    MULTIPLE_CONSTRAINTS = "Multiple constraints"
    POST_PROCESSING = "Post processing"
    TEMPORAL_REASONING = "Temporal reasoning"
    NUMERICAL_REASONING = "Numerical reasoning"
    TABULAR_REASONING = "Tabular reasoning"

@define
class Row:
    id:int
    prompt:str
    answer:str
    tags:list[ReasoningType]
    sources:list[str]

@define
class Dataset:
    rows:list[Row]
