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


MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256


@task
def download_dataset(path: str) -> Dataset:
    dataset = load_dataset(path)

    return dataset

@task
def preprocess(
    dataset: Dataset,
    pretrained_model_name_or_path: str,
    src_lang: str,
    tgt_lang: str,
    preprocess_fun: Callable
):
    tokenizer = M2M100Tokenizer.from_pretrained(
        pretrained_model_name_or_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    tokenized_dataset = dataset.map(preprocess_fun, batched=True)
    return tokenized_dataset


@task
def get_model():
    pass

@task(limits=Resources())  # TODO: change to gpu in production
def train(
    src_lang: str,
    tgt_lang: str,
    tokenized_dataset: Dataset,
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
    pretrained_model_name_or_path: str="facebook/m2m100_418M",
):
    model = M2M100ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    tokenizer = M2M100Tokenizer.from_pretrained(
        pretrained_model_name_or_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model()

@workflow
def adapt_model(pretrained_model_name_or_path: str, hyperparameters):
    # put all the tasks here
    pass
