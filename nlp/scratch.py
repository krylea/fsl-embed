from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, PreTrainedTokenizerFast
import evaluate
import numpy as np

from nlp.utils import train_tokenizer, build_model

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3, 'filter'=lambda ex: ex['label'] != -1},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2},
    'imdb': {'keys': ['text'], 'num_labels': 2}
}

def get_tokenize_function(tokenizer, dataset_keys):
    return lambda examples: tokenizer(*[examples[k] for k in dataset_keys if k is not None], padding="max_length", truncation=True)







metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, tokenizer, dataset_name, training_args, metrics_fct=compute_metrics):
    dataset = load_dataset(dataset_name)
    tokenized_dataset = dataset.map(get_tokenize_function(tokenizer, DATASETS[dataset_name]['keys']), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()


def finetune_bert(dataset_name, max_steps=5000, eval_steps=500):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=DATASETS[dataset_name]['num_labels'])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="steps", save_strategy='no', eval_steps=eval_steps, max_steps=max_steps)
    
    train_model(model, tokenizer, dataset_name, training_args)


def train_simple_model(dataset_name, max_steps=5000, eval_steps=500):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = build_model(tokenizer.vocab_size, 512, 1024, 4, 16, 512, 0.1, 'gelu')
    argdict = {
        'output_dir': 'test_trainer',
        'evaluation_strategy': 'steps', 
        'save_strategy': 'no', 
        'eval_steps': eval_steps, 
        'max_steps': max_steps,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
    }
    training_args = TrainingArguments(**argdict)
    train_model(model, tokenizer, dataset_name, training_args)

import copy
def test_holdout(dataset, model, holdout_indices, train_steps, finetune_steps):
    train_data, holdout_data = [], []
    for i, x in enumerate(dataset):
        holdout=False
        for j in holdout_indices:
            if j in x.ids:
                holdout=True
        if holdout:
            holdout_data.append(x)
        else:
            train_data.append(x)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    argdict = {
        'output_dir': 'test_trainer',
        'evaluation_strategy': 'no', 
        'save_strategy': 'no', 
        'max_steps': train_steps,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
    }
    training_args = TrainingArguments(**argdict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    trainer.train()






from nlp.dataset import *
dataset_name='mnli'
dataset= load_dataset(dataset_name)['train']
vocab_size=20000
top_words=2000
dataset_keys = DATASETS[dataset_name]['keys']
tokenizer = train_tokenizer(dataset, vocab_size, dataset_keys=dataset_keys)
occs = get_occurrences(dataset, tokenizer, dataset_keys, sparse=True)

