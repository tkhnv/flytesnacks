import pandas as pd
from flytekit import workflow
from flytekit.types.directory import FlyteDirectory
from datasets import Dataset

try:
    from compare_systems import compare_systems
    from custom_types import DatasetWithMetadata, Metric, EvaluateReturnType
    from download_dataset import download_dataset
    from filter_length_ratio import filter_length_ratio
    from get_model import get_model, get_tokenizer
    from tokenize import tokenize
    from translate import translate
    from detokenize import detokenize
    from evaluate import evaluate
    from train import train_model
except ImportError:
    from .compare_systems import compare_systems
    from .custom_types import DatasetWithMetadata, Metric, EvaluateReturnType
    from .download_dataset import download_dataset
    from .filter_length_ratio import filter_length_ratio
    from .get_model import get_model, get_tokenizer
    from .tokenize_step import tokenize
    from .translate import translate
    from .detokenize import detokenize
    from .evaluate import evaluate
    from .train import train_model


@workflow
def translate_and_evaluate(
    tokenized_dataset: DatasetWithMetadata, model: FlyteDirectory, tokenizer: FlyteDirectory
) -> EvaluateReturnType:
    translated = translate(tokenized_dataset, model)
    detokenized = detokenize(translated, tokenizer)
    score = evaluate(detokenized, Metric.bleu, {"trust_remote_code": True})
    Metric.chrf
    return score


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
    tokenized_test_dataset = tokenize(test_dataset, tokenizer)
    trained_model = train_model(model, tokenizer, tokenized_train_dataset, {"max_steps": 2})
    score_base = translate_and_evaluate(tokenized_test_dataset, model, tokenizer)
    score_trained = translate_and_evaluate(tokenized_test_dataset, trained_model, tokenizer)
    _ = compare_systems(score_base, score_trained)  # currently nothing is returned
    return score_trained


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
