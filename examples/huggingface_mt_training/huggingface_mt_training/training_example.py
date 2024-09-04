from typing import Callable

from flytekit import task, workflow, Resources
from datasets import load_dataset, Dataset
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import HfApi


MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
HF_TOKEN = None  # use it if you want to fetch a private dataset


@task
def download_dataset(
    dataset_path: str,
    config_name: str,
    src_lang: str,
    tgt_lang: str,
    **load_dataset_kwargs,
) -> Dataset:
    # TODO: consider if this really necessary, we can just load the dataset and use
    # it as is
    def get_features_keys() -> tuple:
        # we expect the dataset to have a "translation" key with two languages
        hf_api = HfApi(token=HF_TOKEN)
        metadata = hf_api.dataset_info(dataset_path)
        try:
            configs = metadata.card_data.dataset_info
        except AttributeError:
            # if there is no metadata, assume that
            # the data are in the correct format already
            return src_lang, tgt_lang
        assert len(configs) != 0

        config = filter(lambda c: c["config_name"] == config_name, configs)
        if len(config) == 0:
            config = configs[0]
        translation_feature_dtype = filter(
            lambda f: f["name"] == "translation",
            config["features"],
        )[0]
        languages = tuple(translation_feature_dtype.values())
        assert len(languages) == 2

        return languages

    s, t = get_features_keys()

    # load the dataset and convert it to unified format
    # of {"translation": {`src_lang`: str, `tgt_lang`: str}}
    dataset = load_dataset(dataset_path, config_name, **load_dataset_kwargs)
    # rename columns
    dataset = dataset.map(
        lambda example: {
            "translation": {
                src_lang: example["translation"][s],
                tgt_lang: example["translation"][t],
            }
    }).select_columns([src_lang, tgt_lang])
    return dataset


@task
def preprocess(dataset: Dataset, preprocess_fun: Callable = lambda e: e, **map_kwargs):
    tokenized_dataset = dataset.map(preprocess_fun, batched=True, **map_kwargs)
    return tokenized_dataset

# TODO: IMPLEMENT ME
@task
def train(
    src_lang: str,
    tgt_lang: str,
    dataset: Dataset,
    model,
    tokenizer,
    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
    ),
):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model()

@workflow
def adapt_model(pretrained_model_name_or_path: str, hyperparameters):
    # put all the tasks here
    dataset = download_dataset("wmt14", "cs-en")
    dataset = preprocess(dataset, lambda e: e, {})


