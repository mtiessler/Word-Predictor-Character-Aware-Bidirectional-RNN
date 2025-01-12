import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
import re
import torch.nn as nn
from argparse import ArgumentParser
import matplotlib.pyplot as plt

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
        cleaned_line = clean_text(line)
        word_counter.update(tokenizer(cleaned_line))
        char_counter.update(tokenize_by_char(cleaned_line))

    word_vocab = {word: idx for idx, (word, _) in enumerate(word_counter.items(), start=2)}
    word_vocab["<unk>"] = 1  # Reserved for unknown words
    char_vocab = {char: idx for idx, (char, _) in enumerate(char_counter.items(), start=2)}
    char_vocab["<unk>"] = 1  # Reserved for unknown characters

    return word_vocab, char_vocab, tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, word_vocab, char_vocab, tokenizer, max_word_len=10, max_seq_len=50):
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
            cleaned_text = clean_text(text)  # Use a different variable name
            word_tokens = [self.word_vocab.get(token, 1) for token in self.tokenizer(cleaned_text)]
            if len(word_tokens) > 1:
                data.append(word_tokens[:self.max_seq_len])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_seq = self.data[idx]
        original_text = self.texts[idx]
        tokens = self.tokenizer(original_text)[:len(word_seq)]

        char_seq = [
            [self.char_vocab.get(char, 1) for char in list(token)] for token in tokens
        ]

        char_seq = [
            chars[:self.max_word_len] + [0] * (self.max_word_len - len(chars))
            for chars in char_seq
        ]

        char_seq_tensor = torch.tensor(char_seq, dtype=torch.long) if char_seq else torch.zeros((1, self.max_word_len), dtype=torch.long)

        return torch.tensor(word_seq[:-1]), torch.tensor(word_seq[1:]), char_seq_tensor

def collate_fn(batch):
    inputs, targets, char_inputs = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    max_seq_len = inputs.size(1)
    max_word_len = max((x.size(1) if x.dim() > 1 else 0) for x in char_inputs)

    char_inputs_padded = torch.zeros((len(char_inputs), max_seq_len, max_word_len), dtype=torch.long)

    for i, char_seq in enumerate(char_inputs):
        if char_seq.dim() == 1:
            char_seq = torch.zeros((1, max_word_len), dtype=torch.long)
        seq_len = char_seq.size(0)
        char_inputs_padded[i, :seq_len, :] = char_seq[:max_seq_len, :]

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
        char_inputs = torch.clamp(char_inputs, min=0, max=self.char_embedding.num_embeddings - 1)
        char_inputs = char_inputs.view(-1, max_word_len)
        char_embeds = self.char_embedding(char_inputs)
        _, (char_hidden, _) = self.char_lstm(char_embeds)
        char_hidden = char_hidden[-1].view(batch_size, seq_len, -1)
        word_embeds = self.word_embedding(word_inputs)
        combined_embeds = torch.cat((word_embeds, char_hidden), dim=-1)
        lstm_out, _ = self.lstm(combined_embeds)
        logits = self.fc(lstm_out)
        return logits

def save_results_to_csv(results, file_name):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy"])
        writer.writerows(results)

def plot_results(file_name):
    import pandas as pd

    data = pd.read_csv(file_name)
    plt.figure()

    plt.plot(data["Epoch"], data["Loss"], label="Loss")
    plt.plot(data["Epoch"], data["Accuracy"], label="Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()

def train_model(model, train_loader, optimizer, device, epochs, csv_file):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    results = []

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (word_inputs, targets, char_inputs) in enumerate(train_loader):
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)
            optimizer.zero_grad()
            outputs = model(word_inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
            total_loss += loss.item()

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_accuracy = correct_predictions / total_predictions * 100
        avg_loss = total_loss / len(train_loader)
        results.append([epoch + 1, avg_loss, train_accuracy])
        print(f"Epoch {epoch + 1}: Avg Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    save_results_to_csv(results, csv_file)

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch_idx, (word_inputs, targets, char_inputs) in enumerate(val_loader):
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)
            outputs = model(word_inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Validation Step [{batch_idx}/{len(val_loader)}], Loss: {loss.item():.4f}")

    perplexity = torch.exp(torch.tensor(total_loss / len(val_loader)))
    accuracy = correct_predictions / total_predictions * 100
    print(f"Evaluation: Perplexity: {perplexity.item():.4f}, Accuracy: {accuracy:.2f}%")
    return perplexity.item(), accuracy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/wikiText-2", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--csv_file", type=str, default="training_results.csv", help="CSV file to save training results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Vocabulary and Tokenizer...")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer(args.data_dir)
    print(f"Word Vocab Size: {len(word_vocab)}, Char Vocab Size: {len(char_vocab)}")

    print("Preparing Data Loaders...")
    train_loader, val_loader = get_dataloaders(args.data_dir, word_vocab, char_vocab, tokenizer, args.batch_size)

    print("Initializing Model...")
    model = LSTMWithCacheAndChar(
        len(word_vocab), len(char_vocab), 128, 32, 256, 64, 2, cache_size=500, max_word_len=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training...")
    train_model(model, train_loader, optimizer, device, args.epochs, args.csv_file)

    print("Evaluating the Model...")
    perplexity, accuracy = evaluate_model(model, val_loader, device)

    print("Final Evaluation Results:")
    print(f"Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%")

    print("Plotting Training Results...")
    plot_results(args.csv_file)
