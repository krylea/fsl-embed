from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, PreTrainedTokenizerFast, BertConfig, BertForSequenceClassification
import evaluate
import numpy as np

from nlp.dataset import NLPDataset, DATASETS
from nlp.models import BertForSequenceClassificationWrapper
from nlp.training_utils import VQTrainer
from embed import SymbolicEmbeddingsGumbel, SymbolicEmbeddingsVQ

import argparse
import copy

#parser = argparse.ArgumentParser()
#parser.add_argument('run_name', type=str)
#parser.add_argument('--use_sym', action='store_true')

#args = parser.parse_args()

voc_size = 20000
top_words = 2000
n_symbols = 2000
pattern_length = 8
latent_size = 512
hidden_size = 1024
num_layers = 4
num_heads = 16
max_length = 128
dropout = 0.1
activation_fct = 'gelu'
use_sym = 'none'
beta=1.
lr=5e-5
num_train_epochs=6.0
batch_size=32
dataset_name='mnli'
num_labels=DATASETS[dataset_name]['num_labels']

def build_simple_model(vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, symbolic_embeds=None):
    config = BertConfig(vocab_size, latent_size, num_layers, num_heads, hidden_size, activation_fct, dropout, dropout, max_length, num_labels=num_labels)
    if symbolic_embeds is None:
        model = BertForSequenceClassification(config)
    else:
        model = BertForSequenceClassificationWrapper(config, symbolic_embeds)
    return model

dataset = NLPDataset.process_dataset(dataset_name, voc_size, top_words)
tokenizer = dataset.tokenizer

sym = None
trainer_cls=Trainer
if use_sym == 'gumbel':
    sym = SymbolicEmbeddingsGumbel(dataset.vocab_size, n_symbols, pattern_length, latent_size // pattern_length)
elif use_sym == 'vq':
    sym = SymbolicEmbeddingsVQ(dataset.vocab_size, n_symbols, pattern_length, latent_size // pattern_length, beta)
    trainer_cls=VQTrainer

model = build_simple_model(voc_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, sym)


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def train(model, dataset, train_steps, eval_every=500, batch_size=32, test_frac=0.1):
    data_collator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="steps", 
        eval_steps=eval_every,
        save_strategy='no', 
        learning_rate=lr, 
        max_steps=train_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )
    train_dataset, test_dataset = dataset.split(test_frac=test_frac)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    trainer.train()
    acc = trainer.evaluate()['eval_accuracy']
    return acc

def fewshot(model, dataset, holdout_words, train_steps, finetune_steps, **kwargs):
    id_dataset, ood_datasets = dataset.pivot(holdout_words)
    train_acc = train(model, id_dataset, train_steps, **kwargs)
    accs = {}
    for word_idx, word_dataset in ood_datasets.items():
        finetune_model = copy.deepcopy(model)
        eval_acc = train(finetune_model, word_dataset, finetune_steps, eval_every=125, test_frac=0.5, **kwargs)
        print("%s Accuracy: %f" % (dataset.tokenizer.convert_ids_to_tokens([word_idx])[0], eval_acc))
        del finetune_model
        accs[word_idx] = eval_acc
    return accs


train(model, dataset, 2000)

'''
occs = dataset.occs
counts = occs.sum(dim=0)
_, sorted_indices = counts.sort(descending=True)
ordered_inds = [37, 106, 87, 98, 55]
holdout_inds = sorted_indices[ordered_inds]
accs = fewshot(model, dataset, holdout_inds, 2000, 500)

words = [dataset.tokenizer.convert_ids_to_tokens([word_idx])[0] for word_idx in holdout_inds]
'''


