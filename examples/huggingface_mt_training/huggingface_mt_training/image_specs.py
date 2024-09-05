from flytekit import ImageSpec

# AutoModelForSeq2SeqLM requires torch to be installed
transformers_image_spec = ImageSpec(
    packages=["transformers", "torch", "datasets", "flytekitplugins-huggingface", "sentencepiece"],
    registry="localhost:30000",
)