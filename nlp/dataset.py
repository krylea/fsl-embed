
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from utils import inverse_permutation

from transformers import PreTrainedTokenizerFast

import datasets

import torch

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3, 'filter':lambda ex: ex['label'] != -1},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2},
    'imdb': {'keys': ['text'], 'num_labels': 2},
    'sst2': {'keys': ['sentence'], 'num_labels': 2}
}

TOKENS = {
    'pad_token': "[PAD]",
    'cls_token': "[CLS]",
    'sep_token': "[SEP]",
    'unk_token': "[UNK]",
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
    N_seqs = len(dataset)
    N_words = len(tokenizer.get_vocab())
    occurences = torch.zeros(N_seqs, N_words, dtype=torch.bool)
    for i, ex in enumerate(dataset):
        for k in dataset_keys:
            tokenized_example = tokenizer.encode(ex[k])
            for idx in tokenized_example.ids:
                occurences[i, idx] = True
    return occurences

def filter_by_top_words(dataset, tokenizer, n_words, dataset_keys=["text"], occs=None):
    if occs == None:
        occs = get_occurrences(dataset, tokenizer, dataset_keys)
    counts = occs.sum(dim=0)
    _, sorted_indices = counts.sort(descending=True)
    #top_words = sorted_indices[:n_words]
    bottom_words = sorted_indices[n_words:]
    id_indices = (occs[:,bottom_words].sum(dim=1) == 0).nonzero().squeeze(1)
    return dataset.select(id_indices.tolist())

def tokenize_dataset(dataset, tokenizer, dataset_keys):
    def get_tokenize_function(tokenizer, dataset_keys):
        return lambda examples: tokenizer(*[examples[k] for k in dataset_keys if k is not None], padding="max_length", truncation=True)
    tokenize_fct = get_tokenize_function(tokenizer, dataset_keys)
    tokenized_dataset = dataset.map(tokenize_fct, batched=True)
    return tokenized_dataset

class NLPDataset():
    @classmethod
    def process_dataset(cls, dataset_name, vocab_size, top_words):
        dataset= datasets.load_dataset(dataset_name)['train']
        if 'filter' in DATASETS[dataset_name]:
            dataset = dataset.filter(DATASETS[dataset_name]['filter'])
        dataset_keys = DATASETS[dataset_name]['keys']
        tokenizer = train_tokenizer(dataset, vocab_size, dataset_keys=dataset_keys)
        occs = get_occurrences(dataset, tokenizer, dataset_keys)
        if top_words > 0:
            dataset = filter_by_top_words(dataset, tokenizer, top_words, dataset_keys, occs)
        final_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **TOKENS)
        tokenized_dataset = tokenize_dataset(dataset, final_tokenizer, dataset_keys)
        
        return cls(dataset_name, tokenized_dataset, final_tokenizer, tokenizer.get_vocab_size(), occs)

    def __init__(self, dataset_name, dataset, tokenizer, vocab_size, occs):
        self.dataset_name = dataset_name
        self.dataset_keys = DATASETS[dataset_name]['keys']
        self.num_labels = DATASETS[dataset_name]['num_labels']

        self.dataset = dataset
        self.N = len(self.dataset)
        self.vocab_size = vocab_size
        
        self.tokenizer = tokenizer
        self.occs = occs
        self.index_map = inverse_permutation(self.dataset['idx'])

    def _index_map(self, indices):
        return [self.index_map[x] if x < self.N and self.index_map[x] >= 0]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.dataset[i]

    def pivot(self, word_indices):
        id_indices = (self.occs[:,word_indices].sum(dim=1) == 0).nonzero().squeeze(1)
        #ood_indices = (occs[:,word_indices].sum(dim=1)).nonzero().squeeze(1)

        ood_indices_by_word = {}
        for idx in word_indices:
            ood_indices_by_word[idx]= self.occs[:,idx].nonzero().squeeze(1)

        return self.partition(id_indices), {k:self.partition(v) for k,v in ood_indices_by_word.items()}

    def partition(self, indices):
        subset = self.dataset.select(self._index_map(indices))
        return NLPDataset(self.dataset_name, subset, self.tokenizer, self.vocab_size, self.occs)
        #ood_indices = torch.tensor([x for x in self.dataset['idx'] if x not in indices])
        #ood_dataset = self.dataset.select(self.index_map[ood_indices].tolist())

    def split(self, test_frac=0.1):
        split_dataset = self.dataset.train_test_split(test_size=test_frac)
        return [NLPDataset(self.dataset_name, split_dataset[split], self.tokenizer, self.vocab_size, self.occs) for split in split_dataset]


        




