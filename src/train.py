import torch
import torch.nn as nn
from model import LSTMWithCacheAndChar
from dataset import get_dataloaders, load_vocab_and_tokenizer
from argparse import ArgumentParser


def train_model(model, train_loader, optimizer, device, epochs, use_sampling=False):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for word_inputs, targets, char_inputs in train_loader:
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)
            optimizer.zero_grad()
            outputs = model(word_inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for word_inputs, targets, char_inputs in val_loader:
            word_inputs, targets, char_inputs = word_inputs.to(device), targets.to(device), char_inputs.to(device)
            outputs = model(word_inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(val_loader)))
    print(f"Validation Perplexity: {perplexity.item()}")
    return perplexity.item()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_caching", action="store_true", help="Enable caching")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer()
    train_loader, val_loader = get_dataloaders(word_vocab, char_vocab, tokenizer, args.batch_size)

    model = LSTMWithCacheAndChar(
        len(word_vocab), len(char_vocab), 128, 32, 256, 64, 2, cache_size=500, max_word_len=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_loader, optimizer, device, args.epochs, use_sampling=args.use_caching)
    evaluate_model(model, val_loader, device)
