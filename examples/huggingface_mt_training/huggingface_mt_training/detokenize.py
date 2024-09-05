from flytekit import task, StructuredDataset
from flytekit.types.directory import FlyteDirectory
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

from .custom_types import DatasetWithMetadata
from .image_specs import transformers_image_spec


@task(container_image=transformers_image_spec)
def detokenize(
    dataset_and_languages: DatasetWithMetadata,
    tokenizer_path: FlyteDirectory,
) -> DatasetWithMetadata:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.path)
    dataset = dataset_and_languages.dataset
    # we expect the dataset to have keys "labels" and "input_ids"
    hf_dataset = Dataset.from_pandas(dataset.open(pd.DataFrame).all())
    hf_dataset = hf_dataset.map(
        lambda batch: {"detokenized_labels": tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)},
        batched=True,
    )
    hf_dataset = hf_dataset.map(
        lambda batch: {"detokenized_input_ids": tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)},
        batched=True,
    )
    return DatasetWithMetadata(
        StructuredDataset(dataframe=hf_dataset),
        dataset_and_languages.source_language,
        dataset_and_languages.target_language,
    )


