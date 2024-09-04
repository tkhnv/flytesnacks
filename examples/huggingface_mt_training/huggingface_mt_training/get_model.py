import flytekit
from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory

try:
    from .image_specs import transformers_image_spec
except ImportError:
    from image_specs import transformers_image_spec


# Increase the RAM required for the task, the model is loaded in memory
@task(container_image=transformers_image_spec, limits=Resources(mem="5G"), requests=Resources(mem="4.5G"))
def get_model(pretrained_model_name: str) -> FlyteDirectory:
    """
    Downloads the model from the Hugging Face model hub and saves it to a directory.
    Args:
        pretrained_model_name: The name of the pretrained model to download.
    Returns:
        FlyteDirectory: The directory containing the model. Each run returns a different directory.
    """
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)

    working_dir = flytekit.current_context().working_directory
    model_dir = f"{working_dir}/model"
    model.save_pretrained(model_dir)
    return FlyteDirectory(path=model_dir)


# Increase the RAM required for the task, the model is loaded in memory
@task(container_image=transformers_image_spec)
def get_tokenizer(pretrained_model_name: str) -> FlyteDirectory:
    """
    Downloads the tokenizer of the model from the Hugging Face model hub and saves it to a directory.
    Args:
        pretrained_model_name: The name of the pretrained model's tokenizer to download.
    Returns:
        FlyteDirectory: The directory containing the model. Each run returns a different directory.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    working_dir = flytekit.current_context().working_directory
    tokenizer_dir = f"{working_dir}/tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    return FlyteDirectory(path=tokenizer_dir)


@workflow
def wf() -> FlyteDirectory:
    """Declare workflow called `wf`."""
    model_name = "facebook/m2m100_418M"
    result1 = get_model(model_name)
    print(result1)
    # We could increase the limits and requires for the second task
    result2 = get_model(model_name)  # .with_overrides(limits=Resources(mem="7G"))
    print(result2)
    tokenizer_dir = get_tokenizer(model_name)
    # result1 and result2 are different directories
    return result1


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
