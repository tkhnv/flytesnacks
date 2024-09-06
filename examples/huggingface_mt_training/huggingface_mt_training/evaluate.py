from flytekit import task

from .custom_types import DatasetWithMetadata, Metric, EvaluateReturnType


metric_name_to_score_map: dict[Metric, str] = {
    "bleu": "bleu",
    "chrf": "score",
}


@task
def evaluate(
    dataset: DatasetWithMetadata,
    metric_name: Metric,
    load_metric_kwargs: dict = {},
) -> EvaluateReturnType:
    from datasets import load_metric, Dataset
    import pandas as pd

    metric = load_metric(metric_name, **load_metric_kwargs)
    structured_dataset = dataset.dataset
    hf_dataset = Dataset.from_pandas(structured_dataset.open(pd.DataFrame).all())
    score = metric.compute(predictions=hf_dataset["detokenized"], references=hf_dataset["target"])[
        metric_name_to_score_map[metric_name]
    ]
    return EvaluateReturnType(score=score)
