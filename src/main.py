import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import re
import csv
from argparse import ArgumentParser
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Utility: Clean and tokenize text
def clean_text(text):
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()


def build_vocab(data_dir, train_texts):
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()

    # Read texts from data_dir or use train_texts if provided
    if data_dir:
        with open(os.path.join(data_dir, "train.txt"), "r") as f:
            for line in f:
                counter.update(tokenizer(clean_text(line)))
    else:
        for line in train_texts:
            counter.update(tokenizer(clean_text(line)))

    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), start=2)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab, tokenizer


class SimpleDataset(Dataset):
    def __init__(self, texts, vocab, tokenizer, max_seq_len=10):
        self.texts = [clean_text(line) for line in texts]
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = [self.vocab.get(token, 1) for token in self.tokenizer(self.texts[idx])]
        if len(tokens) < 2:
            tokens = [0, 0]  # Ensure valid input-target pairs
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


# Simple LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)


# Training Loop
def train_model(model, dataloader, optimizer, device, epochs, csv_file):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    results = []

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"  Processing Batch {batch_idx + 1}/{len(dataloader)}...")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            print(f"    Forward Pass Complete: Outputs Shape: {outputs.shape}")

            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            print(f"    Reshaped Outputs: {outputs.shape}, Targets: {targets.shape}")

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"    Computed Loss: {loss.item():.4f}")

            # Backward pass
            loss.backward()
            print(f"    Backward Pass Complete.")

            # Update weights
            optimizer.step()
            print(f"    Optimizer Step Complete.")

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)
            batch_accuracy = 100.0 * batch_correct / batch_total
            print(f"    Batch Accuracy: {batch_accuracy:.2f}%")

            # Update metrics
            correct += batch_correct
            total += batch_total
            total_loss += loss.item()

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * correct / total
        results.append([epoch + 1, avg_loss, perplexity, accuracy])
        print(f"Epoch {epoch + 1}/{epochs} Summary: "
              f"Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%")

    # Save results to CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Perplexity", "Accuracy"])
        writer.writerows(results)

    print(f"Training Complete. Results saved to {csv_file}.")

# Evaluation
def evaluate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    print("Starting Evaluation...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"  Processing Batch {batch_idx + 1}/{len(dataloader)}...")
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            print(f"    Forward Pass Complete: Outputs Shape: {outputs.shape}")

            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            print(f"    Reshaped Outputs: {outputs.shape}, Targets: {targets.shape}")

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            print(f"    Computed Loss: {loss.item():.4f}")

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)
            batch_accuracy = 100.0 * batch_correct / batch_total
            print(f"    Batch Accuracy: {batch_accuracy:.2f}%")

            # Update metrics
            correct += batch_correct
            total += batch_total

    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = 100.0 * correct / total
    print(f"Evaluation Complete: Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%")
    return perplexity, accuracy




def plot_training_results(csv_file):
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Plot accuracy and perplexity over epochs
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(data['Epoch'], data['Accuracy'], marker='o', label='Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # Perplexity
    plt.subplot(1, 2, 2)
    plt.plot(data['Epoch'], data['Perplexity'], marker='o', label='Perplexity', color='orange')
    plt.title('Perplexity Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Function
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--csv_file", type=str, default="training_results.csv", help="CSV file to save training results")
    args = parser.parse_args()

    if args.data_dir:
        print("Using custom dataset...")
        train_texts = open(os.path.join(args.data_dir, "train.txt"), "r").readlines()
        val_texts = open(os.path.join(args.data_dir, "test.txt"), "r").readlines()
    else:
        print("Using Hugging Face's official wikitext-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        val_texts = dataset["validation"]["text"]

    # Prepare vocabulary and datasets
    vocab, tokenizer = build_vocab(args.data_dir, train_texts)
    train_dataset = SimpleDataset(train_texts, vocab, tokenizer)
    val_dataset = SimpleDataset(val_texts, vocab, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = SimpleLSTM(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train and evaluate
    results = train_model(model, train_loader, optimizer, device, args.epochs, args.csv_file)
    perplexity, accuracy = evaluate_model(model, val_loader, device)
    print(f"Final Results: Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%")

    # Visualize results
    plot_training_results(args.csv_file)
