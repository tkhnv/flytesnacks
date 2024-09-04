from typing import Callable

from flytekit import task, workflow, Resources
from datasets import load_dataset, Dataset
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
HF_TOKEN = None  # use it if you want to fetch a private dataset


@task
def preprocess(dataset: Dataset, preprocess_fun: Callable = lambda e: e, map_kwargs: dict = {}):
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
    dataset, language_pair = download_dataset("wmt14", "cs-en")
    dataset = preprocess(dataset, lambda e: e, {})
