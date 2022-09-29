from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

from nlp.utils import train_tokenizer, build_model

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2},
    'imdb': {'keys': ['text'], 'num_labels': 2}
}

def get_tokenize_function(dataset_keys):
    return lambda examples: tokenizer(*[examples[k] for k in dataset_keys if k is not None], padding="max_length", truncation=True)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, tokenizer, dataset_name, training_args, metrics_fct=compute_metrics):
    dataset = datasets.load_dataset(dataset_name)
    tokenized_dataset = dataset.map(get_tokenize_function(DATASET_KEYS[dataset_name]), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()


def finetune_bert(dataset_name):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    train_model(model, tokenizer, dataset_name, training_args)




