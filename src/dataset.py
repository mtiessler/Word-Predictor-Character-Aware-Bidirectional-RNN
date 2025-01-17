import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def tokenize_by_char(text):
    return list(text)


def load_vocab_and_tokenizer(train_texts):
    tokenizer = lambda x: x.split()  # Simple whitespace tokenizer

    word_counter = Counter()
    char_counter = Counter()

    for line in train_texts:
        clean_line = clean_text(line)
        if clean_line:  # Skip empty lines
            word_counter.update(tokenizer(clean_line))
            char_counter.update(tokenize_by_char(clean_line))

    # Add special tokens and build vocabularies
    word_vocab = {"<pad>": 0, "<unk>": 1}
    word_vocab.update({word: idx for idx, (word, _) in enumerate(word_counter.items(), start=2)})

    char_vocab = {"<pad>": 0, "<unk>": 1}
    char_vocab.update({char: idx for idx, (char, _) in enumerate(char_counter.items(), start=2)})

    return word_vocab, char_vocab, tokenizer


def load_text_datasets():
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()]

    print(f"Train texts: {len(train_texts)}, Validation texts: {len(val_texts)}")
    return train_texts, val_texts


class TextDataset(Dataset):
    def __init__(self, texts, word_vocab, char_vocab, tokenizer, max_word_len=10, max_seq_len=50):
        self.texts = [clean_text(text) for text in texts if text.strip()]
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for text in self.texts:
            word_tokens = [self.word_vocab.get(token, 1) for token in self.tokenizer(text)]
            if word_tokens:  # Keep sequences with at least one token
                if len(word_tokens) < self.max_seq_len:
                    word_tokens += [0] * (self.max_seq_len - len(word_tokens))  # Pad to max_seq_len
                data.append(word_tokens[:self.max_seq_len])  # Truncate if longer
        print(f"Prepared {len(data)} samples for the dataset.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_seq = self.data[idx]
        char_seq = [
            [self.char_vocab.get(char, 1) for char in list(str(token))] for token in word_seq
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

    char_inputs = pad_sequence(
        [pad_sequence(x, batch_first=True, padding_value=0) for x in char_inputs],
        batch_first=True,
        padding_value=0
    )

    return inputs, targets, char_inputs
