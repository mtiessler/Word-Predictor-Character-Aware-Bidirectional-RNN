import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import load_vocab_and_tokenizer, load_text_datasets, TextDataset, collate_fn
from model import LSTMWithCacheAndChar
from train_eval import train_model, evaluate_model, plot_training_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--csv_file", type=str, default="training_results.csv", help="CSV file to save training results")
    parser.add_argument("--max_word_len", type=int, default=10, help="Maximum word length for character sequences")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Maximum sequence length for word sequences")

    args = parser.parse_args()

    print("Loading dataset...")
    train_texts, val_texts = load_text_datasets()

    print("Loading vocabularies and tokenizer...")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer(train_texts)



    print("Preparing datasets and dataloaders...")
    train_dataset = TextDataset(
        train_texts, word_vocab,
        char_vocab,
        tokenizer,
        max_word_len=args.max_word_len,
        max_seq_len=args.max_seq_len
    )
    val_dataset = TextDataset(
        val_texts, word_vocab,
        char_vocab,
        tokenizer,
        max_word_len=args.max_word_len,
        max_seq_len=args.max_seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = LSTMWithCacheAndChar(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        word_embed_dim=128,
        char_embed_dim=64,
        hidden_dim=256,
        char_hidden_dim=128,
        num_layers=2,
        cache_size=100
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    train_model(model, train_loader, optimizer, device, args.epochs, args.csv_file)

    print("Evaluating model...")
    perplexity, accuracy = evaluate_model(model, val_loader, device)
    print(f"Final Results: Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%")

    print("Plotting training results...")
    plot_training_results(args.csv_file)
