from flytekit import task

from .image_specs import transformers_image_spec
from .custom_types import DatasetWithMetadata, Metric, EvaluateReturnType


metric_name_to_score_map: dict[Metric, str] = {
    Metric.bleu: "bleu",
    Metric.chrf: "score",
}


@task(container_image=transformers_image_spec)
def evaluate(
    dataset: DatasetWithMetadata,
    metric_name: Metric,
    load_metric_kwargs: dict = {},
) -> EvaluateReturnType:
    from datasets import load_metric, Dataset
    import pandas as pd
    metric = load_metric(metric_name_to_score_map[metric_name], **load_metric_kwargs)
    structured_dataset = dataset.dataset
    hf_dataset = Dataset.from_pandas(structured_dataset.open(pd.DataFrame).all())
    # TODO: this doesn't crash, but the score is 0. Find out what exactly we need to pass as input to the metric
    score = metric.compute(predictions=[[h] for h in hf_dataset["detokenized"]], references=[[[r]] for r in hf_dataset["target"]])[
        metric_name_to_score_map[metric_name]
    ]
    return EvaluateReturnType(score=score)
