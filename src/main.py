import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
import re
import torch.nn as nn
from argparse import ArgumentParser


# Data preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def tokenize_by_char(text):
    return list(text)


def load_vocab_and_tokenizer(data_dir):
    tokenizer = get_tokenizer("basic_english")

    def read_file(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.readlines()

    train_path = os.path.join(data_dir, "train.txt")
    word_counter = Counter()
    char_counter = Counter()

    for line in read_file(train_path):
        clean_line = clean_text(line)
        word_counter.update(tokenizer(clean_line))
        char_counter.update(tokenize_by_char(clean_line))

    word_vocab = {word: idx for idx, (word, _) in enumerate(word_counter.items(), start=1)}
    char_vocab = {char: idx for idx, (char, _) in enumerate(char_counter.items(), start=1)}

    return word_vocab, char_vocab, tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, word_vocab, char_vocab, tokenizer, max_word_len=10, max_seq_len=50):
        # Load the text lines
        with open(file_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]

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
        word_seq = self.data[idx]  # Word indices
        original_text = self.texts[idx]  # The original text string
        tokens = self.tokenizer(original_text)  # Tokenize the original text into words

        # Ensure the token list is limited to the length of the word_seq
        tokens = tokens[:len(word_seq)]

        # Generate character sequences from the tokens
        char_seq = [
            [self.char_vocab.get(char, 0) for char in list(token)] for token in tokens
        ]

        # Pad or truncate each character sequence
        char_seq = [
            chars[:self.max_word_len] + [0] * (self.max_word_len - len(chars))
            for chars in char_seq
        ]

        # Filter out empty character sequences
        if not char_seq:
            char_seq = [[0] * self.max_word_len]

        # Convert to tensor and ensure 2D shape
        char_seq_tensor = torch.tensor(char_seq, dtype=torch.long)

        # Debugging: Log the shape of char_seq_tensor
        print(f"char_seq_tensor shape (index {idx}): {char_seq_tensor.shape}")

        return torch.tensor(word_seq[:-1]), torch.tensor(word_seq[1:]), char_seq_tensor


def collate_fn(batch):
    inputs, targets, char_inputs = zip(*batch)

    # Pad the word sequences
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Determine the maximum sequence length from `inputs`
    max_seq_len = inputs.size(1)
    max_word_len = max((x.size(1) if x.dim() > 1 else 0) for x in char_inputs)

    # Create a tensor to hold all character sequences
    char_inputs_padded = torch.zeros((len(char_inputs), max_seq_len, max_word_len), dtype=torch.long)

    for i, char_seq in enumerate(char_inputs):
        if char_seq.dim() == 1:  # Handle edge case for empty or malformed sequences
            char_seq = torch.zeros((1, max_word_len), dtype=torch.long)

        # Debugging: Log the shape of each `char_seq`
        print(f"char_seq shape before padding (batch {i}): {char_seq.shape}")

        seq_len = char_seq.size(0)
        char_inputs_padded[i, :seq_len, :] = char_seq[:max_seq_len, :]

    # Debugging: Log the final shape of padded char_inputs
    print(f"char_inputs_padded shape: {char_inputs_padded.shape}")

    return inputs, targets, char_inputs_padded

def get_dataloaders(data_dir, word_vocab, char_vocab, tokenizer, batch_size):
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "test.txt")

    train_loader = DataLoader(
        TextDataset(train_path, word_vocab, char_vocab, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextDataset(valid_path, word_vocab, char_vocab, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# Model definition
class LSTMWithCacheAndChar(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, word_embed_dim, char_embed_dim, hidden_dim, char_hidden_dim, num_layers, cache_size, max_word_len):
        super(LSTMWithCacheAndChar, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)

        self.char_lstm = nn.LSTM(char_embed_dim, char_hidden_dim, batch_first=True)

        self.lstm = nn.LSTM(word_embed_dim + char_hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, word_vocab_size)

        self.cache = {}
        self.cache_size = cache_size

    def forward(self, word_inputs, char_inputs):
        batch_size, seq_len, max_word_len = char_inputs.size()

        # Clamp char_inputs to ensure indices are within valid range
        char_inputs = torch.clamp(char_inputs, min=0, max=self.char_embedding.num_embeddings - 1)

        # Reshape for character-level LSTM
        char_inputs = char_inputs.view(-1, max_word_len)
        char_embeds = self.char_embedding(char_inputs)
        _, (char_hidden, _) = self.char_lstm(char_embeds)

        # Reshape back to batch and sequence level
        char_hidden = char_hidden[-1].view(batch_size, seq_len, -1)

        # Word embeddings
        word_embeds = self.word_embedding(word_inputs)

        # Concatenate word and character embeddings
        combined_embeds = torch.cat((word_embeds, char_hidden), dim=-1)

        # LSTM and fully connected layers
        lstm_out, _ = self.lstm(combined_embeds)
        logits = self.fc(lstm_out)

        # Update cache
        for idx in range(batch_size):
            seq_key = word_inputs[idx].cpu().numpy().tobytes()
            if seq_key not in self.cache:
                self.cache[seq_key] = logits[idx]
                if len(self.cache) > self.cache_size:
                    self.cache.pop(next(iter(self.cache)))

        return logits


# Training and evaluation
def train_model(model, train_loader, optimizer, device, epochs):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        for batch_idx, (word_inputs, targets, char_inputs) in enumerate(train_loader):
            # Move data to device
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)

            # Debugging: Log tensor shapes and max indices
            print(f"\nBatch {batch_idx + 1}:")
            print(f"word_inputs shape: {word_inputs.shape}")
            print(f"targets shape: {targets.shape}")
            print(f"char_inputs shape: {char_inputs.shape}")
            max_index = char_inputs.max().item()
            print(f"Max index in char_inputs: {max_index}, Char vocab size: {model.char_embedding.num_embeddings}")

            # Check for invalid indices
            if max_index >= model.char_embedding.num_embeddings:
                print("Error: Index in char_inputs exceeds char_vocab size!")
                return

            optimizer.zero_grad()
            outputs = model(word_inputs, char_inputs)

            # Debugging: Log outputs shape
            print(f"outputs shape: {outputs.shape}")

            # Flatten for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            print(f"outputs flattened shape: {outputs.shape}")
            print(f"targets flattened shape: {targets.shape}")

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"Batch Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")



def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for batch_idx, (word_inputs, targets, char_inputs) in enumerate(val_loader):
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)

            # Debugging: Log tensor shapes during evaluation
            print(f"\nValidation Batch {batch_idx + 1}:")
            print(f"word_inputs shape: {word_inputs.shape}")
            print(f"targets shape: {targets.shape}")
            print(f"char_inputs shape: {char_inputs.shape}")

            outputs = model(word_inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    perplexity = torch.exp(torch.tensor(total_loss / len(val_loader)))
    print(f"Validation Perplexity: {perplexity.item()}")
    return perplexity.item()

# Main script
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/wikiText-2", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer(args.data_dir)
    train_loader, val_loader = get_dataloaders(args.data_dir, word_vocab, char_vocab, tokenizer, args.batch_size)

    model = LSTMWithCacheAndChar(
        len(word_vocab), len(char_vocab), 128, 32, 256, 64, 2, cache_size=500, max_word_len=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_loader, optimizer, device, args.epochs)
    evaluate_model(model, val_loader, device)
