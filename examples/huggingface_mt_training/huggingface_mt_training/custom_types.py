from dataclasses import dataclass
from enum import Enum

from flytekit.types.structured.structured_dataset import StructuredDataset
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class DatasetWithMetadata(DataClassJSONMixin):
    dataset: StructuredDataset
    source_language: str
    target_language: str

@dataclass
class EvaluateReturnType:
    score: float

class Metric(Enum):
    bleu = "bleu"
    chrf = "chrf"
