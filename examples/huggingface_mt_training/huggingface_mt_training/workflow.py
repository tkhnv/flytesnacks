import pandas as pd
from flytekit import workflow
from datasets import Dataset

try:
    from custom_types import DatasetWithMetadata
    from download_dataset import download_dataset
    from filter_length_ratio import filter_length_ratio
    from get_model import get_model, get_tokenizer
    from tokenize import tokenize
    from translate import translate
    from detokenize import detokenize
    from evaluate import evaluate
except ImportError:
    from .custom_types import DatasetWithMetadata
    from .download_dataset import download_dataset
    from .filter_length_ratio import filter_length_ratio
    from .get_model import get_model, get_tokenizer
    from .tokenize import tokenize
    from .translate import translate
    from .detokenize import detokenize
    from .evaluate import evaluate


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
    translated_dataset = detokenize(translated_dataset, tokenizer)
    # add detokenized translation column to the original dataset
    # TODO: maybe we should do this some other way
    original_hf_dataset = Dataset.from_pandas(dataset.open(pd.DataFrame).all())
    translated_hf_dataset = detokenize(translated_dataset, tokenizer)
    original_hf_dataset = original_hf_dataset.add_column("detokenized", translated_hf_dataset["detokenized"])
    score = evaluate(
        DatasetWithMetadata(
            original_hf_dataset, source_language=dataset.source_language, target_language=dataset.source_language
        ),
        "bleu",
    )
    return translated_dataset


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
