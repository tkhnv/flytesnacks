from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .custom_types import DatasetWithMetadata
    from .image_specs import transformers_image_spec
    from .download_dataset import download_dataset
    from .get_model import get_model, get_tokenizer
    from .tokenize import tokenize
except ImportError:
    from custom_types import DatasetWithMetadata
    from image_specs import transformers_image_spec
    from download_dataset import download_dataset
    from get_model import get_model, get_tokenizer
    from tokenize import tokenize


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
    from torch.utils.data import DataLoader

    model_path.download()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    hf_dataset = Dataset.from_pandas(dataset.dataset.open(pd.DataFrame).all())
    hf_dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)

    translated_dataset = []

    # TODO fix batching - we need to pad somehow
    # for batch in DataLoader(hf_dataset, batch_size=batch_size):
    for batch in DataLoader(hf_dataset, batch_size=1):
        print(batch)
        translated = model.generate(
            batch["input_ids"],
            max_length=max_target_length,
            num_beams=beam_size,
            decoder_start_token_id=model.config.pad_token_id,
        )

        translated_dataset.append(translated)

    hf_dataset.add_column("translated", translated_dataset)
    return DatasetWithMetadata(
        StructuredDataset(dataframe=hf_dataset.to_pandas()), dataset.source_language, dataset.target_language
    )


@workflow
def wf() -> DatasetWithMetadata:
    """Declare workflow called `wf`."""
    dataset = download_dataset("wmt14", "cs-en", {"split": "test"})
    tokenizer = get_tokenizer("facebook/m2m100_418M")
    dataset = tokenize(dataset, tokenizer)

    model = get_model("facebook/m2m100_418M")
    translated = translate(dataset, model)
    return translated
