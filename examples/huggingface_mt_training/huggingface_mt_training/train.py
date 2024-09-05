import flytekit
import pandas as pd
from datasets import Dataset
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

try:
    from .image_specs import transformers_image_spec
except ImportError:
    from image_specs import transformers_image_spec
try:
    from .download_dataset import download_dataset
except ImportError:
    from download_dataset import download_dataset
try:
    from .tokenize import tokenize
except ImportError:
    from tokenize import tokenize
try:
    from .get_model import get_model, get_tokenizer
except ImportError:
    from get_model import get_model, get_tokenizer
try:
    from .custom_types import DatasetWithMetadata
except ImportError:
    from custom_types import DatasetWithMetadata


@task(container_image=transformers_image_spec)
def train_model(
    base_model: FlyteDirectory,
    tokenizer: FlyteDirectory,
    tokenized_dataset: DatasetWithMetadata,
) -> FlyteDirectory:
    from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments

    base_model.download()
    tokenizer.download()
    hf_dataset = Dataset.from_pandas(tokenized_dataset.dataset.open(pd.DataFrame).all())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer.path)

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model.path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir="my-model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    working_dir = flytekit.current_context().working_directory
    model_dir = f"{working_dir}/trained_model"
    trainer.save_model(model_dir)
    return FlyteDirectory(path=model_dir)


@workflow
def wf() -> FlyteDirectory:
    model_name = "facebook/m2m100_418M"
    tokenizer = get_tokenizer(model_name)
    tokenized_dataset = tokenize(download_dataset("wmt14", "cs-en", {"split": "test"}), tokenizer)
    base_model = get_model(model_name)
    trained_model = train_model(base_model, tokenizer, tokenized_dataset)
    return trained_model


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() {wf()}")
