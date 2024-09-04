from typing import NamedTuple

from datasets import Dataset, load_dataset
from flytekit import task, workflow

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
HF_TOKEN = None  # use it if you want to fetch a private dataset

DatasetWithMetadata = NamedTuple("DatasetWithMetadata", dataset=Dataset, source_language=str, target_language=str)


@task
def download_dataset(
    dataset_path: str,
    config_name: str,
    load_dataset_kwargs: dict = {},
) -> DatasetWithMetadata:
    # load the dataset and convert it to unified format
    # of {"translation": {`src_lang`: str, `tgt_lang`: str}}
    dataset = load_dataset(dataset_path, config_name, split="test")
    # rename columns
    dataset = dataset.select_columns("translation")
    return DatasetWithMetadata(dataset, *dataset.info.features["translation"].languages)


@workflow
def wf() -> DatasetWithMetadata:
    # put all the tasks here
    dataset = download_dataset("wmt14", "cs-en", {})
    # dataset = preprocess(dataset, lambda e: e, {})
    return dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
