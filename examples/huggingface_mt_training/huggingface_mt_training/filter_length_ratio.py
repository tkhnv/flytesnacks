from flytekit import dynamic, task, workflow
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .custom_types import DatasetWithMetadata
    from .image_specs import transformers_image_spec
    from .download_dataset import download_dataset
except ImportError:
    from custom_types import DatasetWithMetadata
    from image_specs import transformers_image_spec
    from download_dataset import download_dataset


def lengths_are_ok(
    example: dict,
    max_ratio: float = 2.0,
    max_length: int = 200,
    source_key: str = "source",
    target_key: str = "target",
) -> bool:
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


@task(container_image=transformers_image_spec)
def filter_df(
    dataset: StructuredDataset,
    max_ratio: float = 2.0,
    max_length: int = 200,
    source_key: str = "source",
    target_key: str = "target",
) -> StructuredDataset:
    """
    Drops rows that have a length ratio greater than max_ratio or a length greater than max_length.
    Args:
        dataset: Structured dataset to be filtered
        max_ratio: The maximum allowed length ratio of a segment.
        max_length: The maximum allowed length of a segment.
        source_key: Name of the key where source texts are in the dataset.
        target_key: Name of the key where target texts are in the dataset.
    Returns:
        StructuredDataset: Filtered dataset.
    """
    import pandas as pd

    # Convert structured dataset to pandas dataframe
    pd_df = dataset.open(pd.DataFrame).all()
    pd_df["lengths_are_ok"] = pd_df.apply(
        lambda row: lengths_are_ok(row, max_ratio, max_length, source_key, target_key), axis=1
    )
    pd_df = pd_df[pd_df["lengths_are_ok"]]
    pd_df = pd_df.drop(columns=["lengths_are_ok"])
    return StructuredDataset(dataframe=pd_df)


@task(container_image=transformers_image_spec)
def create_output(v: list[StructuredDataset], source_language: str, target_language: str) -> DatasetWithMetadata:
    import pandas as pd

    # Concatenate all partial results into one big dataset
    v = [dataset.open(pd.DataFrame).all() for dataset in v]
    dataset = pd.concat(v)
    return DatasetWithMetadata(StructuredDataset(dataframe=dataset), source_language, target_language)


@dynamic(container_image=transformers_image_spec)
def filter_length_ratio(
    dataset: DatasetWithMetadata,
    max_ratio: float = 2.0,
    max_length: int = 200,
    source_key: str = "source",
    target_key: str = "target",
    chunk_size: int = 1000,
) -> DatasetWithMetadata:
    """Compute the length ratio of each segment in a list of dictionaries.
    Args:
        data: Huggingface dataset.
        max_ratio: The maximum allowed length ratio of a segment.
        max_length: The maximum allowed length of a segment.
        source_key: The key of the source text in the dictionary.
        target_key: The key of the target text in the dictionary.
        chunk_size: The size of the chunks to split the dataset into.
    Returns:
        DatasetWithMetadata: The filtered dataset.
    """
    import pandas as pd

    # Convert to pandas dataframe
    pd_df = dataset.dataset.open(pd.DataFrame).all()

    # Split the df into chunks of 1000 rows
    v = []
    for i in range(0, len(pd_df), chunk_size):
        v.append(
            filter_df(
                StructuredDataset(dataframe=pd_df[i : i + chunk_size]), max_ratio, max_length, source_key, target_key
            )
        )
    return create_output(v, dataset.source_language, dataset.target_language)


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    filtered_dataset = filter_length_ratio(dataset)
    return filtered_dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    import pandas as pd

    res = wf()
    print(res)
    print(f"Running wf() {res.dataset.open(pd.DataFrame).all().shape}")
