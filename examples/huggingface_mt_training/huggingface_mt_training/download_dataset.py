from datasets import load_dataset
from flytekit import task, workflow

try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata

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
    languages = dataset.info.features["translation"].languages
    # Rename columns to source and target
    dataset = (
        dataset.flatten()
        .rename_column(f"translation.{languages[0]}", "source")
        .rename_column(f"translation.{languages[1]}", "target")
    )
    return DatasetWithMetadata(dataset, *languages)


@workflow
def wf() -> DatasetWithMetadata:
    # put all the tasks here
    dataset = download_dataset("wmt14", "cs-en")
    # dataset = preprocess(dataset, lambda e: e, {})
    return dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
