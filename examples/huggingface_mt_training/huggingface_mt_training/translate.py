from flytekit import task
from flytekit.types.directory import FlyteDirectory

try:
    from .types import DatasetWithMetadata
except ImportError:
    from types import DatasetWithMetadata


# create a flyte task to translate a tokenized dataset with M2M100 model
@task
def translate(
    dataset: DatasetWithMetadata,
    model: FlyteDirectory,
    max_target_length: int = 256,
    batch_size: int = 8,
    beam_size: int = 4,
):
    """
    Translate a tokenized dataset using the M2M100 model.
    Args:
        dataset (DatasetWithMetadata): The tokenized dataset to translate.
        model (FlyteDirectory): The directory containing the model.
        max_target_length (int, optional): The maximum length of the target sequence. Defaults to 256.
        batch_size (int, optional): The batch size for translation. Defaults to 8.
        beam_size (int, optional): The beam size for translation. Defaults
    Returns:
        Dataset: The translated dataset.
    """
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    translated_dataset = dataset.dataset.map(
        lambda e: model.generate(
            e["input_ids"],
            max_length=max_target_length,
            num_beams=beam_size,
            decoder_start_token_id=model.config.pad_token_id,
        ),
        batched=True,
        batch_size=batch_size,
    )

    return translated_dataset
