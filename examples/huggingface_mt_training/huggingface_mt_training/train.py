from flytekit.types.structured.structured_dataset import StructuredDataset
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
try:
    from .image_specs import transformers_image_spec
except ImportError:
    from image_specs import transformers_image_spec
try:
    from .download_dataset import download_dataset
except ImportError:
    from download_dataset import download_dataset
try:
    from .get_model import get_model
except ImportError:
    from get_model import get_model
try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata


@task(container_image=transformers_image_spec)
def train_model(
    base_model: FlyteDirectory,
    tokenized_dataset: DatasetWithMetadata,
) -> FlyteDirectory:
    print(tokenized_dataset)
    return base_model


@workflow
def wf() -> FlyteDirectory:
    dataset_and_languages = download_dataset("wmt14", "cs-en")
    base_model = get_model("facebook/m2m100_418M")
    trained_model = train_model(base_model, dataset_and_languages)
    return trained_model


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
