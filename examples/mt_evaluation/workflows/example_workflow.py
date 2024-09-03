"""A basic Flyte project template that uses ImageSpec"""

import typing
from flytekit import task, workflow
try:
  from get_sample_data import get_sample_data
  from sacrebleu_metrics import compute_sacrebleu_metrics
  from comet_metrics import compute_comet_metrics
except:
  from .get_sample_data import get_sample_data
  from .sacrebleu_metrics import compute_sacrebleu_metrics
  from .comet_metrics import compute_comet_metrics
  

@task
def gather_metrics(data: list[dict]) -> dict:
    sacrebleu_metrics = compute_sacrebleu_metrics(data=data)
    comet_metric = compute_comet_metrics(data=data)
    metrics = sacrebleu_metrics.update(comet_metric)
    return metrics
    
@workflow
def wf() -> dict:
    """Declare workflow called `wf`.
    """
    data = get_sample_data()
    metrics = gather_metrics(data=data)
    return metrics

if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
