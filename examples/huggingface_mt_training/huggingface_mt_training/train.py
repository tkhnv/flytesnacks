from typing import NamedTuple

from datasets import Dataset, load_dataset
from flytekit import task, workflow, ImageSpec
from flytekit.types.directory import FlyteDirectory

custom_image = ImageSpec(
    packages=["transformers", "torch", "datasets"],
    registry="localhost:30000",
    base_image="ubuntu:focal"
)


DatasetWithMetadata = NamedTuple("DatasetWithMetadata", dataset=Dataset, source_language=str, target_language=str)

@task(container_image=custom_image)
def train_model(
    base_model: FlyteDirectory,
    tokenized_dataset: Dataset,
) -> FlyteDirectory:
    return base_model


@workflow
def wf() -> FlyteDirectory:
    from .download_dataset import download_dataset
    dataset_and_languages = download_dataset("wmt14", "cs-en")
    from .get_model import  get_model
    base_model = get_model("facebook/m2m100_418M")
    trained_model = train_model(base_model, dataset_and_languages.dataset)
    return trained_model


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
