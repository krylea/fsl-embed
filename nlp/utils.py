
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

import datasets

def batch_iterator(dataset, batch_size=1000, dataset_keys=["text"]):
    for i in range(0, len(dataset), batch_size):
        yield [x for k in dataset_keys for x in dataset[i : i + batch_size][k]]


def train_tokenizer(dataset, vocab_size, batch_size=1000, dataset_keys=["text"]):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
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
        occ_table[:,i][occs_i] = 1

    labels = torch.tensor([x['label'] for x in dataset])
    return occ_table, labels




#snli = datasets.load_dataset("snli", split="train")

from transformers import BertConfig, BertForSequenceClassification

def build_model(vocab_size, latent_size, hidden_size, num_layers, num_heads, max_length, dropout, activation_fct):
    config = BertConfig(vocab_size, latent_size, num_layers, num_heads, hidden_size, activation_fct, dropout, dropout, max_length)
    model = BertForSequenceClassification(config)
    return model
