from datasets import load_dataset
from flytekit import task, workflow

from examples.huggingface_mt_training.huggingface_mt_training.types import DatasetWithMetadata

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
HF_TOKEN = None  # use it if you want to fetch a private dataset


@task
def download_dataset(
    dataset_path: str,
    config_name: str,
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
