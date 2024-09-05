from flytekit import ImageSpec

# AutoModelForSeq2SeqLM requires torch to be installed
transformers_image_spec = ImageSpec(
    packages=["transformers", "torch", "pandas", "datasets", "flytekitplugins-huggingface", "sentencepiece", "pandas", "accelerate"],
    registry="localhost:30000",
)
