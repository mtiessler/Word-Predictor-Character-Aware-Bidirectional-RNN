import os
import csv
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from dataset import load_vocab_and_tokenizer, load_text_datasets, TextDataset, collate_fn
from model import LSTMWithCacheAndChar
from train_eval import train_model, evaluate_model, plot_training_results


def smoke_test():
    print("Running smoke test...")
    # Create synthetic data
    synthetic_texts = ["this is a test", "another test sentence"]
    # Small vocab for testing
    word_vocab = {word: idx for idx, word in enumerate(["<pad>", "this", "is", "a", "test", "another", "sentence"])}
    char_vocab = {char: idx for idx, char in enumerate(list("abcdefghijklmnopqrstuvwxyz"))}

    # Create tokenizer mock
    tokenizer = lambda text: [word_vocab[word] for word in text.split() if word in word_vocab]

    # Prepare dataset and dataloader
    dataset = TextDataset(
        synthetic_texts, word_vocab, char_vocab, tokenizer, max_word_len=5, max_seq_len=10
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = LSTMWithCacheAndChar(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        word_embed_dim=8,
        char_embed_dim=4,
        hidden_dim=16,
        char_hidden_dim=8,
        num_layers=1,
        cache_size=10,
        dropout_rate=0.1,
        l2_lambda=1e-4
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Single forward-backward pass
    for inputs, targets, char_inputs in dataloader:
        inputs, targets, char_inputs = (
            inputs.to(device),
            targets.to(device),
            char_inputs.to(device),
        )

        optimizer.zero_grad()
        outputs = model(inputs, char_inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        # Loss calculation
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")
        break  # Only one iteration for smoke test

    print("Smoke test passed!")


def main():
    """Main function to run the training and evaluation pipeline."""
    # Load environment variables
    load_dotenv()

    # Check if smoke test is enabled
    smoke_test_enabled = os.getenv("SMOKE_TEST", "False").lower() == "true"
    if smoke_test_enabled:
        smoke_test()
        return

    # Read configuration from .env file
    batch_size = int(os.getenv("BATCH_SIZE", 32))
    epochs = int(os.getenv("EPOCHS", 100))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
    csv_file = os.getenv("CSV_FILE", "training_results.csv")
    max_word_len = int(os.getenv("MAX_WORD_LEN", 10))
    max_seq_len = int(os.getenv("MAX_SEQ_LEN", 50))
    word_embed_dim = int(os.getenv("WORD_EMBED_DIM", 128))
    char_embed_dim = int(os.getenv("CHAR_EMBED_DIM", 64))
    hidden_dim = int(os.getenv("HIDDEN_DIM", 256))
    char_hidden_dim = int(os.getenv("CHAR_HIDDEN_DIM", 128))
    num_layers = int(os.getenv("NUM_LAYERS", 2))
    cache_size = int(os.getenv("CACHE_SIZE", 100))
    dropout_rate = float(os.getenv("DROPOUT_RATE", 0.5))
    l2_lambda = float(os.getenv("L2_LAMBDA", 1e-5))

    # Experiment name and paths
    experiment_name = f"experiment_batch{batch_size}_epoch{epochs}_lr{learning_rate}"
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    csv_file_path = os.path.join("results", f"{experiment_name}.csv")

    # Initialize CSV file
    csv_headers = [
        "Epoch",
        "Loss (Cross-Entropy)",
        "Perplexity",
        "Accuracy",
        "Avg Levenshtein",
        "Execution Time (s)",
        "Energy Consumption (J)"
    ]
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

    print("Loading dataset...")
    train_texts, val_texts = load_text_datasets()

    print("Loading vocabularies and tokenizer...")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer(train_texts)

    print("Preparing datasets and dataloaders...")
    train_dataset = TextDataset(
        train_texts, word_vocab,
        char_vocab,
        tokenizer,
        max_word_len=max_word_len,
        max_seq_len=max_seq_len
    )
    val_dataset = TextDataset(
        val_texts, word_vocab,
        char_vocab,
        tokenizer,
        max_word_len=max_word_len,
        max_seq_len=max_seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = LSTMWithCacheAndChar(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        word_embed_dim=word_embed_dim,
        char_embed_dim=char_embed_dim,
        hidden_dim=hidden_dim,
        char_hidden_dim=char_hidden_dim,
        num_layers=num_layers,
        cache_size=cache_size,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    train_model(model, train_loader, optimizer, device, epochs, csv_file_path)

    print("Evaluating model...")
    eval_loss, eval_perplexity, eval_accuracy, eval_edit_distance, eval_time, eval_energy = evaluate_model(model,
                                                                                                           val_loader,
                                                                                                           device)

    # Log evaluation results to CSV
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Evaluation",
            eval_loss,
            eval_perplexity,
            eval_accuracy,
            eval_edit_distance,
            eval_time,
            eval_energy
        ])

    print("Plotting training results...")
    plot_training_results(csv_file_path)


if __name__ == "__main__":
    main()
