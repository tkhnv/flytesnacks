from flytekit import task

try:
    from .custom_types import DatasetWithMetadata, EvaluateReturnType, Metric
    from .image_specs import transformers_image_spec
except ImportError:
    from custom_types import DatasetWithMetadata, EvaluateReturnType, Metric
    from image_specs import transformers_image_spec


metric_name_to_score_map: dict[Metric, str] = {
    Metric.bleu: "sacrebleu",
    Metric.chrf: "chrf",
}


@task(container_image=transformers_image_spec)
def evaluate(
    dataset: DatasetWithMetadata,
    metric_name: Metric,
    load_metric_kwargs: dict = {},
) -> EvaluateReturnType:
    import pandas as pd
    from datasets import Dataset, load_metric

    metric = load_metric(metric_name_to_score_map[metric_name], **load_metric_kwargs)
    structured_dataset = dataset.dataset
    hf_dataset = Dataset.from_pandas(structured_dataset.open(pd.DataFrame).all())
    # TODO: this doesn't crash, but the score is 0. Find out what exactly we need to pass as input to the metric
    score = metric.compute(
        predictions=[[h] for h in hf_dataset["detokenized"]], references=[[[r]] for r in hf_dataset["target"]]
    )["score"]
    return EvaluateReturnType(score=score)
