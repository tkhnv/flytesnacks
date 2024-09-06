from flytekit import ImageSpec

# AutoModelForSeq2SeqLM requires torch to be installed
transformers_image_spec = ImageSpec(
    packages=[
        "transformers",
        "torch",
        "pandas",
        "datasets",
        "flytekitplugins-huggingface",
        "sentencepiece",
        "accelerate",
        "plotly",
    ],
    registry="localhost:30000",
)

plot_image_spec = ImageSpec(
    packages=[
        "pandas",
        "plotly",
    ],
    registry="localhost:30000",
)
