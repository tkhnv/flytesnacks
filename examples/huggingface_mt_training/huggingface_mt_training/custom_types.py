from datasets import Dataset


from typing import NamedTuple


DatasetWithMetadata = NamedTuple("DatasetWithMetadata", dataset=Dataset, source_language=str, target_language=str)
