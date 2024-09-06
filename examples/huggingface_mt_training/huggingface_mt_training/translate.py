from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured.structured_dataset import StructuredDataset

try:
    from .custom_types import DatasetWithMetadata
    from .download_dataset import download_dataset
    from .get_model import get_model, get_tokenizer
    from .image_specs import transformers_image_spec
    from .tokenize_step import tokenize
except ImportError:
    from tokenize import tokenize

    from custom_types import DatasetWithMetadata
    from download_dataset import download_dataset
    from get_model import get_model, get_tokenizer
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
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM

    model_path.download()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    hf_dataset = Dataset.from_pandas(dataset.dataset.open(pd.DataFrame).all())
    hf_dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)
    translated_dataset = []
    # Get the target language token id to make m2m100 translate to target language
    # TODO do this properly, also this doesn't work at the moment
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    # TODO fix batching - we need to pad somehow
    # for batch in DataLoader(hf_dataset, batch_size=batch_size):
    for batch in DataLoader(hf_dataset, batch_size=1):
        translated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=torch.LongTensor(batch["attention_mask"]).unsqueeze(0),
            max_length=max_target_length,
            num_beams=beam_size,
            forced_bos_token_id=tokenizer.lang_code_to_id[dataset.target_language],
        )
        translated_dataset.append(translated[0, :].tolist())

    hf_dataset = hf_dataset.add_column("translated", translated_dataset)
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
