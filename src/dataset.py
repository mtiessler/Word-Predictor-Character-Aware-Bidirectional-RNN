import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import re


def clean_text(text):
    """
    Cleans a given text by converting it to lowercase and removing non-alphanumeric characters.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def tokenize_by_char(text):
    """
    Tokenizes a given text into individual characters.

    Args:
        text (str): The input text.

    Returns:
        list: A list of characters from the input text.
    """
    return list(text)


def load_vocab_and_tokenizer(train_texts):
    """
    Constructs word and character vocabularies from the training texts and a tokenizer.

    Args:
        train_texts (list): List of training text strings.

    Returns:
        tuple: A tuple containing:
            - word_vocab (dict): Vocabulary mapping words to indices.
            - char_vocab (dict): Vocabulary mapping characters to indices.
            - tokenizer (function): A function for tokenizing text into words.
    """
    print("Building vocabularies and tokenizer...")
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

    print(f"Word vocab size: {len(word_vocab)}, Char vocab size: {len(char_vocab)}")
    return word_vocab, char_vocab, tokenizer


def load_text_datasets():
    """
    Loads the WikiText-2 dataset and splits it into training, validation, and test sets.

    Returns:
        tuple: A tuple containing:
            - train_texts (list): List of training text strings.
            - val_texts (list): List of validation text strings.
            - test_texts (list): List of test text strings.
    """
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()]
    test_texts = [text for text in dataset["test"]["text"] if text.strip()]

    print(
        f"Loaded dataset: Train texts = {len(train_texts)}, Validation texts = {len(val_texts)}, Test texts = {len(test_texts)}")
    return train_texts, val_texts, test_texts


class TextDataset(Dataset):
    """
    A custom dataset class for handling text data with word and character tokenization.

    Args:
        texts (list): List of text strings.
        word_vocab (dict): Vocabulary mapping words to indices.
        char_vocab (dict): Vocabulary mapping characters to indices.
        tokenizer (function): A function for tokenizing text into words.
        max_word_len (int): Maximum length of character sequences for words.
        max_seq_len (int): Maximum length of word sequences.
    """
    def __init__(self, texts, word_vocab, char_vocab, tokenizer, max_word_len=10, max_seq_len=50):
        print(f"Initializing dataset with {len(texts)} texts...")
        self.texts = [clean_text(text) for text in texts if text.strip()]
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tokenizer = tokenizer
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.data = self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the dataset by tokenizing text into word sequences and padding/truncating them.

        Returns:
            list: A list of padded and truncated word token sequences.
        """
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
        """
        Retrieves a single data sample at the specified index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: A tuple containing:
                - word_seq (torch.Tensor): Word token sequence (input).
                - target_seq (torch.Tensor): Word token sequence (target for prediction).
                - char_seq (torch.Tensor): Character token sequences for words in the input.
        """
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
    """
    Custom collation function for batching data samples.

    Args:
        batch (list): A list of tuples, where each tuple contains:
            - inputs (torch.Tensor): Input word token sequence.
            - targets (torch.Tensor): Target word token sequence.
            - char_inputs (torch.Tensor): Character token sequences.

    Returns:
        tuple: A tuple containing:
            - inputs (torch.Tensor): Padded input sequences.
            - targets (torch.Tensor): Padded target sequences.
            - char_inputs (torch.Tensor): Padded character token sequences.
    """
    inputs, targets, char_inputs = zip(*batch)

    # Pad sequences for word-level inputs and targets
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Pad character sequences
    char_inputs = pad_sequence(
        [pad_sequence(x, batch_first=True, padding_value=0) for x in char_inputs],
        batch_first=True,
        padding_value=0
    )

    print("Batch prepared: inputs shape =", inputs.shape, ", targets shape =", targets.shape)
    return inputs, targets, char_inputs
