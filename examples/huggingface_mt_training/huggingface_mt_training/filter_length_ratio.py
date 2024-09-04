import pandas as pd
from flytekit import task, workflow
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata


def lengths_are_ok(
    example: dict,
    max_ratio: float = 2.0,
    max_length: int = 200,
    source_key: str = "source",
    target_key: str = "target",
) -> bool:
    # for example in examples:
    source = example[source_key]
    target = example[target_key]
    if (
        len(source) / len(target) > max_ratio
        or len(target) / len(source) > max_ratio
        or len(source) > max_length
        or len(target) > max_length
    ):
        return False
    return True


@task
def compute_length_ratio(
    dataset: DatasetWithMetadata,
    max_ratio: float = 2.0,
    max_length: int = 200,
    source_key: str = "source",
    target_key: str = "target",
) -> DatasetWithMetadata:
    """Compute the length ratio of each segment in a list of dictionaries.
    Args:
        data: Huggingface dataset.
        max_ratio: The maximum allowed length ratio of a segment.
        max_length: The maximum allowed length of a segment.
        source_key: The key of the source text in the dictionary.
        target_key: The key of the target text in the dictionary.

    """
    pd_df = dataset.dataset.open(pd.DataFrame).all()
    pd_df["lengths_are_ok"] = pd_df.apply(
        lambda row: lengths_are_ok(row, max_ratio, max_length, source_key, target_key), axis=1
    )
    pd_df = pd_df[pd_df["lengths_are_ok"]]
    pd_df = pd_df.drop(columns=["lengths_are_ok"])
    return DatasetWithMetadata(StructuredDataset(dataframe=pd_df), dataset.source_language, dataset.target_language)


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    try:
        from download_dataset import download_dataset
    except ImportError:
        from .download_dataset import download_dataset
    dataset = download_dataset("wmt14", "cs-en")
    filtered_dataset = compute_length_ratio(dataset)
    return filtered_dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    res = wf()
    print(res)
    print(f"Running wf() {res.dataset.open(pd.DataFrame).all().shape}")
