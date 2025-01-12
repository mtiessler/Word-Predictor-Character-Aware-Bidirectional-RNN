import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def tokenize_by_char(text):
    return list(text)


def load_vocab_and_tokenizer():
    tokenizer = get_tokenizer("basic_english")
    train_iter = WikiText2(split="train")

    word_counter = Counter()
    char_counter = Counter()

    for line in train_iter:
        clean_line = clean_text(line)
        word_counter.update(tokenizer(clean_line))
        char_counter.update(tokenize_by_char(clean_line))

    word_vocab = Counter(word_counter)
    char_vocab = Counter(char_counter)

    return word_vocab, char_vocab, tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, word_vocab, char_vocab, tokenizer, max_word_len=10, max_seq_len=50):
        self.texts = [clean_text(text) for text in texts]
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for text in self.texts:
            word_tokens = [self.word_vocab.get(token, 0) for token in self.tokenizer(text)]
            if len(word_tokens) > 1:
                data.append(word_tokens[:self.max_seq_len])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_seq = self.data[idx]
        char_seq = [
            [self.char_vocab.get(char, 0) for char in list(token)] for token in word_seq
        ]
        char_seq = [
            chars[:self.max_word_len] + [0] * (self.max_word_len - len(chars))
            for chars in char_seq
        ]
        return torch.tensor(word_seq[:-1]), torch.tensor(word_seq[1:]), torch.tensor(char_seq[:-1])


def collate_fn(batch):
    inputs, targets, char_inputs = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    char_inputs = [torch.tensor(x) for x in char_inputs]
    char_inputs = pad_sequence(char_inputs, batch_first=True, padding_value=0)
    return inputs, targets, char_inputs


def get_dataloaders(word_vocab, char_vocab, tokenizer, batch_size):
    train_iter = WikiText2(split="train")
    valid_iter = WikiText2(split="valid")
    train_loader = DataLoader(
        TextDataset(train_iter, word_vocab, char_vocab, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextDataset(valid_iter, word_vocab, char_vocab, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
