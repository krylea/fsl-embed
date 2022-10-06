from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, PreTrainedTokenizerFast, BertConfig, BertForSequenceClassification
import evaluate
import numpy as np

from nlp.dataset import NLPDataset
from nlp.models import BertForSequenceClassificationWrapper
from embed import SymbolicEmbeddingsGumbel

import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('run_name', type=str)
#parser.add_argument('--use_sym', action='store_true')

#args = parser.parse_args()

voc_size = 20000
top_words = 2000
n_symbols = 500
pattern_length = 8
latent_size = 512
hidden_size = 1024
num_layers = 4
num_heads = 16
max_length = 128
dropout = 0.1
activation_fct = 'gelu'
use_sym=True

def build_simple_model(vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, symbolic_embeds=None):
    config = BertConfig(vocab_size, latent_size, num_layers, num_heads, hidden_size, activation_fct, dropout, dropout, max_length)
    if symbolic_embeds is None:
        model = BertForSequenceClassification(config)
    else:
        model = BertForSequenceClassificationWrapper(config, symbolic_embeds)
    return model

train_dataset, val_dataset, test_dataset = NLPDataset.process_splits("sst2", voc_size, top_words)
tokenizer = train_dataset.tokenizer

sym = SymbolicEmbeddingsGumbel(train_dataset.vocab_size, n_symbols, pattern_length, latent_size // pattern_length) if use_sym else None
model = build_simple_model(voc_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, sym)



metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", save_strategy='no')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()

