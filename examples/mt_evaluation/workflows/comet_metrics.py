from flytekit import task, ImageSpec

custom_image = ImageSpec(
    packages=["unbabel-comet"],
    registry="localhost:30000",
)

@task(container_image=custom_image)
def compute_comet_metrics(data: list[dict], source_key: str="source", target_key: str="target", mt_key: str = "mt", batch_size:int=2) -> dict:
    """Compute the length ratio of each segment in a list of dictionaries.
    Args:
        data: A list of segment, has to contain source_key and target_key.
        source_key: The key of the source text in the dictionary.
        target_key: The key of the target text in the dictionary.
        mt_key: The key of the MT in the dictionary.
    """
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    comet_inputs = [{"src": instance[source_key], "mt": instance[mt_key], "ref": instance[target_key]} for instance in data]
    predictions = model.predict(comet_inputs, batch_size=batch_size)[0] #, gpus=self._gpus)
    return {
        "comet": sum(predictions) / len(predictions),
    }
