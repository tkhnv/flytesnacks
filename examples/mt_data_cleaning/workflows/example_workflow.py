"""A basic Flyte project template that uses ImageSpec"""

import typing
from flytekit import task, workflow
try:
  from get_sample_data import get_sample_data
  from filter_length_ratio import compute_length_ratio
except:
  from .get_sample_data import get_sample_data
  from .filter_length_ratio import compute_length_ratio
  
@workflow
def wf() -> list[dict]:
    """Declare workflow called `wf`.
    """
    data = get_sample_data()
    filtered_data = compute_length_ratio(data=data)
    return filtered_data


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
