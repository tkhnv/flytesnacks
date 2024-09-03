
from flytekit import task, workflow, ImageSpec, Resources
from flytekit.types.directory import FlyteDirectory
import flytekit
# AutoModelForSeq2SeqLM requires torch to be installed
custom_image = ImageSpec(
    packages=["transformers", "torch"],
    registry="localhost:30000",
)

# Increase the RAM required for the task, the model is loaded in memory
@task(container_image=custom_image, limits=Resources(mem="5G"), 
    requests=Resources(mem="4.5G",),)
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


@workflow
def wf() -> FlyteDirectory:
    """Declare workflow called `wf`.
    """
    result1 = get_model("facebook/m2m100_418M")
    print(result1)
    # We could increase the limits and requires for the second task
    result2 = get_model("facebook/m2m100_418M") #.with_overrides(limits=Resources(mem="7G"))
    print(result2)
    # result1 and result2 are different directories
    return result1

if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")