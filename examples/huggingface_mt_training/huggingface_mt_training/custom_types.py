from dataclasses import dataclass
from typing import Literal

from flytekit.types.structured.structured_dataset import StructuredDataset
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class DatasetWithMetadata(DataClassJSONMixin):
    dataset: StructuredDataset
    source_language: str
    target_language: str

Metric = Literal["bleu", "chrf"]
