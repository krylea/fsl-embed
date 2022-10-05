
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

import datasets

import torch

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3, 'filter':lambda ex: ex['label'] != -1},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2},
    'imdb': {'keys': ['text'], 'num_labels': 2},
    'sst-2': {'keys': ['sentence'], 'num_labels': 2}
}

def batch_iterator(dataset, batch_size=1000, dataset_keys=["text"]):
    for i in range(0, len(dataset), batch_size):
        yield [x for k in dataset_keys for x in dataset[i : i + batch_size][k]]


def train_tokenizer(dataset, vocab_size, batch_size=1000, dataset_keys=["text"]):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace()])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordLevelTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    )   
    tokenizer.train_from_iterator(batch_iterator(dataset, batch_size=batch_size, dataset_keys=dataset_keys), trainer=trainer, length=len(dataset))
    return tokenizer

def count_vocab(dataset, tokenizer, dataset_keys=["text"]):
    occurences = {}
    for i, ex in enumerate(dataset):
        for k in dataset_keys:
            tokenized_example = tokenizer.encode(ex[k])
            for idx in tokenized_example.ids:
                if idx not in occurences:
                    occurences[idx] = set()
                occurences[idx].add(i)
    return occurences

def word_correlations(dataset, tokenizer, dataset_keys=["text"]):
    occurences= count_vocab(dataset, tokenizer, dataset_keys)
    N_seqs = len(dataset)
    N_words = len(tokenizer.get_vocab())
    occ_table = torch.zeros(N_seqs, N_words, dtype=torch.int)
    for i, occs_i in occurences.items():
        occ_table[:,i][list(occs_i)] = 1
    labels = torch.tensor([x['label'] for x in dataset])
    return occ_table, labels

def tokenize_dataset(dataset, tokenizer, dataset_keys):
    tokenize_fct = lambda examples: tokenizer.encode_batch(*[examples[k] for k in dataset_keys if k is not None])
    return dataset.map(tokenize_fct, batched=True)

def filter_task_by_top_words(dataset, tokenizer, n_words, dataset_keys=["text"], counts=None):
    if counts == None:
        occs = count_vocab(dataset, tokenizer, dataset_keys)
        counts = {k:len(v) for k,v in occs.items()}
    sorted_indices = sorted(counts, key=counts.get, reverse=True)
    top_words = sorted_indices[:n_words]
    def _filter(row):
        for k in dataset_keys:
            tokenized_row = tokenizer.encode(row[k])
            for idx in tokenized_row.ids:
                if idx not in top_words:
                    return False
        return True
    return dataset.filter(_filter)


def process_dataset(dataset_name):
    dataset= datasets.load_dataset(dataset_name)
    if 'filter' in DATASETS[dataset_name]:
        dataset = dataset.filter(DATASETS[dataset_name]['filter'])
    dataset_keys = DATASETS[dataset_name]['keys']
    tokenizer = train_tokenizer(dataset['train'], 20000, dataset_keys=dataset_keys)
    occs = count_vocab(dataset['train'], tokenizer, dataset_keys)
    counts = {k:len(v) for k,v in occs.items()}
    return dataset, tokenizer, counts



#snli = datasets.load_dataset("snli", split="train").filter(lambda ex: ex['label'] != -1)

from transformers import BertConfig, BertForSequenceClassification

def build_model(vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct):
    config = BertConfig(vocab_size, latent_size, num_layers, num_heads, hidden_size, activation_fct, dropout, dropout, max_length)
    model = BertForSequenceClassification(config)
    return model
