import flytekit
from flytekit import ImageSpec, Resources, task, workflow
from flytekit.types.directory import FlyteDirectory

try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata
try:
    from tokenize import tokenize

    from download_dataset import download_dataset
    from filter_length_ratio import filter_length_ratio
    from get_model import get_model, get_tokenizer
    from translate import translate
except ImportError:
    from .download_dataset import download_dataset
    from .filter_length_ratio import filter_length_ratio
    from .get_model import get_model, get_tokenizer
    from .tokenize import tokenize
    from .translate import translate


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    model_name = "facebook/m2m100_418M"
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    filtered_dataset = filter_length_ratio(dataset)
    tokenized_dataset = tokenize(filtered_dataset, tokenizer)
    translated_dataset = translate(tokenized_dataset, model)
    return translated_dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
