from flytekit import task, workflow
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .image_specs import transformers_image_spec
    from .custom_types import DatasetWithMetadata
except ImportError:
    from image_specs import transformers_image_spec
    from custom_types import DatasetWithMetadata


@task(container_image=transformers_image_spec)
def download_dataset(
    dataset_path: str,
    config_name: str,
    load_dataset_kwargs: dict,
) -> DatasetWithMetadata:
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, config_name, **load_dataset_kwargs)
    languages = dataset.info.features["translation"].languages
    # Rename columns to source and target
    dataset = (
        dataset.flatten()
        .take(5)  # TODO remove me for final version
        .rename_column(f"translation.{languages[0]}", "source")
        .rename_column(f"translation.{languages[1]}", "target")
    )
    return DatasetWithMetadata(StructuredDataset(dataframe=dataset), *languages)


@workflow
def wf() -> DatasetWithMetadata:
    # put all the tasks here
    dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    # dataset = preprocess(dataset, lambda e: e, {})
    return dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
