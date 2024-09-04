from flytekit import ImageSpec

# AutoModelForSeq2SeqLM requires torch to be installed
transformers_image_spec = ImageSpec(
    packages=["transformers", "torch"],
    registry="localhost:30000",
)
