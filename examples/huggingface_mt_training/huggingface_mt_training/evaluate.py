from flytekit import task
from datasets import load_metric, Dataset
import pandas as pd

from .custom_types import DatasetWithMetadata, Metric, EvaluateReturnType


metric_name_to_score_map: dict[Metric, str] = {
    "bleu": "bleu",
    "chrf": "score",
}


@task
def evaluate(
    dataset: DatasetWithMetadata,
    metric: Metric,
    load_metric_kwargs: dict={},
) -> EvaluateReturnType:
    metric = load_metric(metric, **load_metric_kwargs)
    hf_dataset = Dataset.from_pandas(dataset.open(pd.DataFrame).all())
    score = metric.compute(
        predictions=hf_dataset["detokenized"], references=hf_dataset["target"]
    )[metric_name_to_score_map[metric]]
    return EvaluateReturnType(score=score)

