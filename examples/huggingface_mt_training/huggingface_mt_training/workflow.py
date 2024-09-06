import pandas as pd
from flytekit import workflow
from datasets import Dataset

try:
    from custom_types import DatasetWithMetadata, Metric
    from download_dataset import download_dataset
    from filter_length_ratio import filter_length_ratio
    from get_model import get_model, get_tokenizer
    from tokenize import tokenize
    from translate import translate
    from detokenize import detokenize
    from evaluate import evaluate
    from train import train_model
except ImportError:
    from .custom_types import DatasetWithMetadata, Metric
    from .download_dataset import download_dataset
    from .filter_length_ratio import filter_length_ratio
    from .get_model import get_model, get_tokenizer
    from .tokenize import tokenize
    from .translate import translate
    from .detokenize import detokenize
    from .evaluate import evaluate
    from .train import train_model


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    model_name = "facebook/m2m100_418M"
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    test_dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    train_dataset = download_dataset("wmt14", "cs-en", {"split": "train"})
    filtered_dataset = filter_length_ratio(train_dataset)
    tokenized_train_dataset = tokenize(filtered_dataset, tokenizer)
    trained_model = train_model(model, tokenizer, tokenized_train_dataset, {"max_steps": 2})
    tokenized_test_dataset = tokenize(test_dataset, tokenizer)
    translated_dataset_base = translate(tokenized_test_dataset, model)
    detokenized_dataset_base = detokenize(translated_dataset_base, tokenizer)
    score_base = evaluate(detokenized_dataset_base, Metric.bleu, {"trust_remote_code": True})
    translated_dataset_trained = translate(tokenized_test_dataset, trained_model)
    detokenized_dataset_trained = detokenize(translated_dataset_trained, tokenizer)
    score_trained = evaluate(detokenized_dataset_trained, Metric.bleu, {"trust_remote_code": True})
    return score_trained


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
