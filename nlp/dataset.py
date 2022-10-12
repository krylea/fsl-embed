
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

import torch
import torch.sparse

import datasets
import tqdm

DATASETS = {
    'snli': {'keys': ['premise', 'hypothesis'], 'num_labels': 3, 'filter':lambda ex: ex['label'] != -1, 'sparse': False},
    'mrpc': {'keys': ['sentence1', 'sentence2'], 'num_labels': 2, 'sparse': False},
    'imdb': {'keys': ['text'], 'num_labels': 2, 'sparse': False},
    'sst2': {'keys': ['sentence'], 'num_labels': 2, 'sparse': False},
    'mnli': {'keys': ['premise', 'hypothesis'], 'num_labels':3, 'parent': 'glue', 'sparse': True}
}

TOKENS = {
    'pad_token': "[PAD]",
    'cls_token': "[CLS]",
    'sep_token': "[SEP]",
    'unk_token': "[UNK]",
}


def load_dataset(dataset_name):
    assert dataset_name in DATASETS
    dataset_args = DATASETS[dataset_name]
    if 'parent' in dataset_args:
        dataset = datasets.load_dataset(dataset_args['parent'], dataset_name)
    else:
        dataset = datasets.load_dataset(dataset_name)
    if 'filter' in dataset_args:
        dataset = dataset.filter(dataset_args['filter'])
    return dataset


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

def get_occurrences(dataset, tokenizer, dataset_keys=["text"], sparse=False):
    N_seqs = len(dataset)
    N_words = len(tokenizer.get_vocab())
    if not sparse:
        occurences = torch.zeros(N_seqs, N_words, dtype=torch.int)
        for i, ex in tqdm.tqdm(enumerate(dataset)):
            for k in dataset_keys:
                tokenized_example = tokenizer.encode(ex[k])
                for idx in tokenized_example.ids:
                    occurences[i, idx] = True
        counts = occurences.sum(dim=0)
    else: 
        indices = [[], []]
        values = []
        for i, ex in tqdm.tqdm(enumerate(dataset)):
            for k in dataset_keys:
                tokenized_example = tokenizer.encode(ex[k])
                for idx in tokenized_example.ids:
                    indices[0].append(i)
                    indices[1].append(idx)
                    values.append(1)
        occurences = torch.sparse_coo_tensor(indices, values, (N_seqs,N_words))
        counts = torch.sparse.sum(occurences, dim=0).to_dense()
    return occurences, counts


def filter_by_top_words(dataset, tokenizer, n_words, occs, counts, dataset_keys=["text"], sparse=False):
    _, sorted_indices = counts.sort(descending=True)
    bottom_words = sorted_indices[n_words:]
    if sparse:
        bottom_counts = torch.sparse.sum(occs.index_select(1,bottom_words), dim=1).to_dense()
        id_indices = (bottom_counts == 0).nonzero().squeeze(1)
    else:
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
        dataset= load_dataset(dataset_name)['train']
        dataset_keys = DATASETS[dataset_name]['keys']
        sparse = DATASETS[dataset_name]['sparse']
        tokenizer = train_tokenizer(dataset, vocab_size, dataset_keys=dataset_keys)
        occs, counts = get_occurrences(dataset, tokenizer, dataset_keys, sparse=sparse)
        if top_words > 0:
            dataset = filter_by_top_words(dataset, tokenizer, top_words, occs, counts, dataset_keys, sparse=sparse)
        final_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **TOKENS)
        tokenized_dataset = tokenize_dataset(dataset, final_tokenizer, dataset_keys)
        
        return cls(dataset_name, tokenized_dataset, final_tokenizer, tokenizer.get_vocab_size(), occs, counts, sparse)

    def __init__(self, dataset_name, dataset, tokenizer, vocab_size, occs, counts, sparse=False):
        self.dataset_name = dataset_name
        self.dataset_keys = DATASETS[dataset_name]['keys']
        self.num_labels = DATASETS[dataset_name]['num_labels']

        self.dataset = dataset
        self.N = len(self.dataset)
        self.vocab_size = vocab_size
        
        self.tokenizer = tokenizer
        self.occs = occs
        self.counts = counts
        self.index_map = inverse_permutation(self.dataset['idx'])

        self.sparse = sparse

    def _index_map(self, indices):
        return [self.index_map[x] for x in indices if x < self.index_map.size(0) and self.index_map[x] >= 0]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.dataset[i]

    def pivot(self, word_indices):
        if self.sparse:
            return self._pivot_sparse(word_indices)
        else:
            return self._pivot(word_indices)

    def _pivot(self, word_indices):
        id_indices = (self.occs[:,word_indices].sum(dim=1) == 0).nonzero().squeeze(1)

        ood_indices_by_word = {}
        for idx in word_indices:
            ood_indices_by_word[idx.item()]= self.occs[:,idx].nonzero().squeeze(1)

        return self.partition(id_indices), {k:self.partition(v) for k,v in ood_indices_by_word.items()}
    
    def _pivot_sparse(self, word_indices):
        word_counts = torch.sparse.sum(occs.index_select(1,bottom_words), dim=1).to_dense()
        id_indices = (word_counts == 0).nonzero().squeeze(1)

        ood_indices_by_word = {}
        for idx in word_indices:
            ood_indices_by_word[idx.item()]= self.occs.index_select(1, idx).to_dense().nonzero().squeeze(1)

        return self.partition(id_indices), {k:self.partition(v) for k,v in ood_indices_by_word.items()}

    def partition(self, indices):
        subset = self.dataset.select(self._index_map(indices))
        return NLPDataset(self.dataset_name, subset, self.tokenizer, self.vocab_size, self.occs, self.counts)
        #ood_indices = torch.tensor([x for x in self.dataset['idx'] if x not in indices])
        #ood_dataset = self.dataset.select(self.index_map[ood_indices].tolist())

    def split(self, test_frac=0.1):
        split_dataset = self.dataset.train_test_split(test_size=test_frac)
        return [NLPDataset(self.dataset_name, split_dataset[split], self.tokenizer, self.vocab_size, self.occs, self.counts) for split in split_dataset]


        




