from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured.structured_dataset import StructuredDataset
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

try:
    from .image_specs import transformers_image_spec
    from .custom_types import DatasetWithMetadata
    from .download_dataset import download_dataset
    from .get_model import get_tokenizer
except ImportError:
    from image_specs import transformers_image_spec
    from custom_types import DatasetWithMetadata
    from download_dataset import download_dataset
    from get_model import get_tokenizer

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
HF_TOKEN = None  # use it if you want to fetch a private dataset


@task(container_image=transformers_image_spec)
def tokenize(
    dataset_and_languages: DatasetWithMetadata,
    tokenizer_path: FlyteDirectory,
) -> DatasetWithMetadata:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.path)
    tokenizer.src_lang = dataset_and_languages.source_language
    tokenizer.tgt_lang = dataset_and_languages.target_language
    dataset = dataset_and_languages.dataset
    hf_dataset = Dataset.from_pandas(dataset.open(pd.DataFrame).all())
    hf_dataset = hf_dataset.map(
        lambda batch: tokenizer(
            batch["source"],
            max_length=MAX_INPUT_LENGTH,
            padding=False,
            truncation=True,
        ),
        batched=True,
    )
    labels = hf_dataset.map(
        lambda batch: tokenizer(
            text_target=batch["target"],
            max_length=MAX_TARGET_LENGTH,
            padding=False,
            truncation=True,
        ),
        batched=True,
    )
    hf_dataset = hf_dataset.add_column("labels", labels["input_ids"])

    return DatasetWithMetadata(
        StructuredDataset(dataframe=hf_dataset),
        dataset_and_languages.source_language,
        dataset_and_languages.target_language,
    )


@workflow
def wf() -> DatasetWithMetadata:
    # put all the tasks here
    dataset = download_dataset("wmt14", "cs-en", split="test")
    tokenizer = get_tokenizer("facebook/m2m100_418M")
    dataset = tokenize(dataset, tokenizer)
    return dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
