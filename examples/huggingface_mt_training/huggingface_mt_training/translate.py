from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata

try:
    from .image_specs import transformers_image_spec
except ImportError:
    from image_specs import transformers_image_spec


# translate a tokenized dataset with M2M100 model
@task(container_image=transformers_image_spec, limits=Resources(mem="5G"), requests=Resources(mem="4.5G"))
def translate(
    dataset: DatasetWithMetadata,
    model_path: FlyteDirectory,
    max_target_length: int = 256,
    batch_size: int = 8,
    beam_size: int = 4,
) -> DatasetWithMetadata:
    """
    Translate a tokenized dataset using the M2M100 model.
    Args:
        dataset (DatasetWithMetadata): The tokenized dataset to translate.
        model_path (FlyteDirectory): The directory containing the model.
        max_target_length (int, optional): The maximum length of the target sequence. Defaults to 256.
        batch_size (int, optional): The batch size for translation. Defaults to 8.
        beam_size (int, optional): The beam size for translation. Defaults to 4.
    Returns:
        Dataset: The translated dataset.
    """
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoModelForSeq2SeqLM

    model_path.download()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    hf_dataset = Dataset.from_pandas(dataset.dataset.open(pd.DataFrame).all())

    translated_dataset = hf_dataset.map(
        lambda e: model.generate(
            e["input_ids"],
            max_length=max_target_length,
            num_beams=beam_size,
            decoder_start_token_id=model.config.pad_token_id,
        ),
        batched=True,
        batch_size=batch_size,
    )

    return StructuredDataset(dataframe=translated_dataset.to_pandas())
