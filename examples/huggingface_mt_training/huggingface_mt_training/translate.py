from flytekit import task

from transformers import M2M100ForConditionalGeneration

try:
    from .types import DatasetWithMetadata
except ImportError:
    from types import DatasetWithMetadata


# create a flyte task to translate a tokenized dataset with M2M100 model
@task
def translate(
    dataset: DatasetWithMetadata,
    model: M2M100ForConditionalGeneration,
    max_target_length: int = 256,
    batch_size: int = 8,
):
    """
    Translate a tokenized dataset using the M2M100 model.
    Args:
        dataset (DatasetWithMetadata): The tokenized dataset to translate.
        model (M2M100ForConditionalGeneration): The model to use for translation.
        max_target_length (int, optional): The maximum target length for the model. Defaults to 256.
        batch_size (int, optional): The batch size for translation. Defaults to 8.
    Returns:
        Dataset: The translated dataset.
    """
    # translate the dataset
    translated_dataset = dataset.dataset.map(
        lambda e: model.generate(
            e["input_ids"],
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
            decoder_start_token_id=model.config.pad_token_id,
        ),
        batched=True,
        batch_size=batch_size,
    )

    return translated_dataset
