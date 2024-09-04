from datasets import Dataset
from dataclasses import dataclass


@dataclass
class DatasetWithMetadata:
    dataset: Dataset
    source_language: str
    target_language: str
