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
dataset_name='mnli'
max_size=-1
'''
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
num_labels=DATASETS[dataset_name]['num_labels']
'''

def build_simple_model(vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, symbolic_embeds=None):
    config = BertConfig(vocab_size, latent_size, num_layers, num_heads, hidden_size, activation_fct, dropout, dropout, max_length, num_labels=num_labels)
    if symbolic_embeds is None:
        model = BertForSequenceClassification(config)
    else:
        model = BertForSequenceClassificationWrapper(config, symbolic_embeds)
    return model


def train(model, dataset, train_steps, trainer_cls=Trainer, eval_every=500, batch_size=32, test_frac=0.1, lr=5e-5):
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

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
    initial_acc = trainer.evaluate()['eval_accuracy']
    trainer.train()
    final_acc = trainer.evaluate()['eval_accuracy']
    return initial_acc, final_acc

def fewshot(model, ood_datasets, finetune_steps, lr=5e-5, trainer_cls=Trainer, eval_every=125, freeze=True, **kwargs):
    accs = {}
    for word_idx, word_dataset in ood_datasets.items():
        finetune_model = copy.deepcopy(model)
        if freeze:
            finetune_model.freeze_model()
        initial_acc, eval_acc = train(finetune_model, word_dataset, finetune_steps, eval_every=eval_every, test_frac=0.5, trainer_cls=trainer_cls, **kwargs)
        print("\n%s: \tInitial Accuracy: %f\tFinal Accuracy: %f" % (dataset.tokenizer.convert_ids_to_tokens([word_idx])[0], initial_acc, eval_acc))
        del finetune_model
        accs[word_idx] = (initial_acc, eval_acc)
    return accs


def run_pretrain(dataset, holdout_words, use_sym='none', n_symbols=2000, pattern_length=8, augment_dim=-1, lr=5e-5, train_steps=6000, batch_size=32,
        latent_size = 512, hidden_size = 1024, num_layers = 4, num_heads = 16, max_length = 128, dropout = 0.1, activation_fct = 'gelu', **sym_kwargs):
    sym = None
    trainer_cls=Trainer
    symbol_dim = latent_size // pattern_length if augment_dim <= 0 else (latent_size-augment_dim) // pattern_length
    if use_sym == 'gumbel':
        sym = SymbolicEmbeddingsGumbel(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
    elif use_sym == 'vq':
        sym = SymbolicEmbeddingsVQ(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
        trainer_cls=VQTrainer

    num_labels=dataset.num_labels
    model = build_simple_model(dataset.vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, sym)
    tokenizer = dataset.tokenizer

    id_dataset, ood_datasets = dataset.pivot(holdout_words)

    _, acc = train(model, id_dataset, train_steps, lr=lr, batch_size=batch_size, trainer_cls=trainer_cls)
    print("\n Accuracy: %f" % acc)

    return model, id_dataset, ood_datasets, trainer_cls


def run_train(dataset, use_sym='none', n_symbols=2000, pattern_length=8, augment_dim=-1, lr=5e-5, train_steps=6000, batch_size=32,
        latent_size = 512, hidden_size = 1024, num_layers = 4, num_heads = 16, max_length = 128, dropout = 0.1, activation_fct = 'gelu', **sym_kwargs):
    sym = None
    trainer_cls=Trainer
    symbol_dim = latent_size // pattern_length if augment_dim <= 0 else (latent_size-augment_dim) // pattern_length
    if use_sym == 'gumbel':
        sym = SymbolicEmbeddingsGumbel(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
    elif use_sym == 'vq':
        sym = SymbolicEmbeddingsVQ(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
        trainer_cls=VQTrainer

    num_labels=dataset.num_labels
    model = build_simple_model(dataset.vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, sym)
    tokenizer = dataset.tokenizer

    _, acc = train(model, dataset, train_steps, lr=lr, batch_size=batch_size, trainer_cls=trainer_cls)
    print("\n Accuracy: %f" % acc)

    '''
    if holdout_inds is not None:
        words = [dataset.tokenizer.convert_ids_to_tokens([word_idx])[0] for word_idx in holdout_inds]
        base_acc, fsl_accs = fewshot(model, dataset, holdout_inds, train_steps, 500, lr=lr, batch_size=batch_size, trainer_cls=trainer_cls)

        print("\nID accuracy: %f" % base_acc)
        for k, v in fsl_accs.items():
            print("%s Accuracy: %f" % (dataset.tokenizer.convert_ids_to_tokens([k])[0], v))
    else:
        '''

def run_fewshot(dataset, holdout_words, use_sym='none', n_symbols=2000, pattern_length=8, augment_dim=-1, lr=5e-5, train_steps=6000, finetune_steps=500, batch_size=32,
        latent_size = 512, hidden_size = 1024, num_layers = 4, num_heads = 16, max_length = 128, dropout = 0.1, activation_fct = 'gelu', **sym_kwargs):
    sym = None
    trainer_cls=Trainer
    symbol_dim = latent_size // pattern_length if augment_dim <= 0 else (latent_size-augment_dim) // pattern_length
    if use_sym == 'gumbel':
        sym = SymbolicEmbeddingsGumbel(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
    elif use_sym == 'vq':
        sym = SymbolicEmbeddingsVQ(dataset.vocab_size, n_symbols, pattern_length, symbol_dim, augment_dim=augment_dim, **sym_kwargs)
        trainer_cls=VQTrainer

    num_labels=dataset.num_labels
    model = build_simple_model(dataset.vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct, num_labels, sym)
    tokenizer = dataset.tokenizer

    id_dataset, ood_datasets = dataset.pivot(holdout_words)

    id_acc = _, train(model, id_dataset, train_steps, lr=lr, batch_size=batch_size, trainer_cls=trainer_cls)

    ood_accs = fewshot(model, ood_datasets, finetune_steps, trainer_cls=trainer_cls, lr=lr)

    print("\nID accuracy: %f" % id_acc)
    for k, v in ood_accs.items():
        print("%s: \tInitial Accuracy: %f\tFinal Accuracy: %f" % (dataset.tokenizer.convert_ids_to_tokens([k])[0], *v))

    #return model, id_dataset, ood_datasets, trainer_cls

        

dataset = NLPDataset.process_dataset(dataset_name, voc_size, top_words, max_size=max_size)

counts = dataset.counts
_, sorted_indices = counts.sort(descending=True)
ordered_inds = [37, 106, 87, 98, 55]
holdout_inds = sorted_indices[ordered_inds]


#base_model, id_dataset, ood_datasets, base_trainer_cls = run_pretrain(dataset, use_sym='none')

#vq_model, id_dataset, ood_datasets, vq_trainer_cls = run_pretrain(dataset, use_sym='vq', pattern_length=16, n_symbols=500)

