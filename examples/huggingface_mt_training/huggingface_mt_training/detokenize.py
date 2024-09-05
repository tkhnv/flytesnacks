from flytekit import task, StructuredDataset, workflow
from flytekit.types.directory import FlyteDirectory
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

try:
    from .custom_types import DatasetWithMetadata
    from .image_specs import transformers_image_spec
    from .download_dataset import download_dataset
    from .get_model import get_model, get_tokenizer
    from .tokenize import tokenize
    from .translate import translate
except ImportError:
    from custom_types import DatasetWithMetadata
    from image_specs import transformers_image_spec
    from download_dataset import download_dataset
    from get_model import get_model, get_tokenizer
    from tokenize import tokenize
    from translate import translate


@task(container_image=transformers_image_spec)
def detokenize(
    dataset_and_languages: DatasetWithMetadata,
    tokenizer_path: FlyteDirectory,
) -> DatasetWithMetadata:
    tokenizer_path.download()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.path)
    tokenizer.src_lang = dataset_and_languages.source_language
    tokenizer.tgt_lang = dataset_and_languages.target_language

    dataset = dataset_and_languages.dataset
    # we expect the dataset to have keys "labels" and "input_ids"
    hf_dataset = Dataset.from_pandas(dataset.open(pd.DataFrame).all())
    hf_dataset = hf_dataset.map(
        lambda batch: {"detokenized": tokenizer.batch_decode(batch["translated"], skip_special_tokens=True)},
        batched=True,
    )
    return DatasetWithMetadata(
        StructuredDataset(dataframe=hf_dataset),
        dataset_and_languages.source_language,
        dataset_and_languages.target_language,
    )


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    tokenizer = get_tokenizer("facebook/m2m100_418M")
    dataset = tokenize(dataset, tokenizer)

    model = get_model("facebook/m2m100_418M")
    translated = translate(dataset, model)
    detokenized = detokenize(translated, tokenizer)
    return detokenized
