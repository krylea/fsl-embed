
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast

import datasets

import torch

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3, 'filter':lambda ex: ex['label'] != -1},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2},
    'imdb': {'keys': ['text'], 'num_labels': 2},
    'sst2': {'keys': ['sentence'], 'num_labels': 2}
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

def get_occurrences(dataset, tokenizer, dataset_keys=["text"]):
    occurrences = {}
    for i, ex in enumerate(dataset):
        for k in dataset_keys:
            tokenized_example = tokenizer.encode(ex[k])
            for idx in tokenized_example.ids:
                if idx not in occurrences:
                    occurrences[idx] = set()
                occurrences[idx].add(i)
    return occurrences

def filter_by_top_words(dataset, tokenizer, n_words, dataset_keys=["text"], counts=None):
    if counts == None:
        occs = get_occurrences(dataset, tokenizer, dataset_keys)
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


def process_dataset(dataset_name, vocab_size, top_words):
    dataset= datasets.load_dataset(dataset_name)
    if 'filter' in DATASETS[dataset_name]:
        dataset = dataset.filter(DATASETS[dataset_name]['filter'])
    dataset_keys = DATASETS[dataset_name]['keys']
    tokenizer = train_tokenizer(dataset['train'], vocab_size, dataset_keys=dataset_keys)
    occs = get_occurrences(dataset['train'], tokenizer, dataset_keys)
    counts = {k:len(v) for k,v in occs.items()}
    if top_words > 0:
        dataset = filter_by_top_words(dataset, tokenizer, top_words, dataset_keys, counts)
    
    return dataset, tokenizer, occs


class NLPDataset():
    @classmethod
    def process_splits(cls, dataset_name, vocab_size, top_words):
        dataset= datasets.load_dataset(dataset_name)
        if 'filter' in DATASETS[dataset_name]:
            dataset = dataset.filter(DATASETS[dataset_name]['filter'])
        dataset_keys = DATASETS[dataset_name]['keys']
        tokenizer = train_tokenizer(dataset['train'], vocab_size, dataset_keys=dataset_keys)
        occs = get_occurrences(dataset['train'], tokenizer, dataset_keys)
        counts = {k:len(v) for k,v in occs.items()}
        if top_words > 0:
            dataset = filter_by_top_words(dataset, tokenizer, top_words, dataset_keys, counts)
        
        return [cls(dataset_name, dataset[split], tokenizer, occs) for split in dataset.keys()]

    def __init__(self, dataset_name, dataset, tokenizer, occs):
        self.dataset_name = dataset_name
        self.dataset_keys = DATASETS[dataset_name]['keys']
        self.num_labels = DATASETS[dataset_name]['num_labels']

        self.dataset = dataset
        self.N = len(self.dataset)
        self.vocab_size = tokenizer.get_vocab_size()
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.occs = {k:torch.tensor(list(v)) for k,v in occs.items()}

        
        self.tokenized_dataset = self._tokenize_dataset()

    def _tokenize_dataset(self):
        def get_tokenize_function(tokenizer, dataset_keys):
            return lambda examples: tokenizer(*[examples[k] for k in dataset_keys if k is not None], padding="max_length", truncation=True)
        tokenize_fct = get_tokenize_function(self.tokenizer, self.dataset_keys)
        tokenized_dataset = self.dataset.map(tokenize_fct, batched=True)
        return tokenized_dataset

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.tokenized_dataset[i]

    #def pivot(self)



