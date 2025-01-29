import os
import csv
from torch.utils.data import DataLoader
from dataset import load_vocab_and_tokenizer, load_text_datasets, TextDataset, collate_fn
from model import LSTMWithCacheAndChar
from train_eval import train_model, evaluate_model, plot_training_results, plot_aggregated_results
import torch


def load_config_from_csv(config_file):
    """
    Load experiment configurations from a CSV file.

    Args:
        config_file (str): Path to the CSV file containing key-value pairs of configurations.

    Returns:
        dict: A dictionary containing the experiment configurations with keys and parsed values.
    """
    config = {}
    with open(config_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row["key"]
            value = row["value"]
            if value.isdigit():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value
    return config


def smoke_test():
    """
    Perform a smoke test to verify that the model, dataset, and training loop work correctly.

    This function uses a small synthetic dataset to check the end-to-end workflow of:
        - Dataset creation
        - Model initialization
        - Forward and backward passes
        - Loss calculation and optimization
    """
    print("Running smoke test...")
    synthetic_texts = ["this is a test", "another test sentence"]
    word_vocab = {word: idx for idx, word in enumerate(["<pad>", "this", "is", "a", "test", "another", "sentence"])}
    char_vocab = {char: idx for idx, char in enumerate(list("abcdefghijklmnopqrstuvwxyz"))}
    tokenizer = lambda text: [word_vocab[word] for word in text.split() if word in word_vocab]

    dataset = TextDataset(
        synthetic_texts, word_vocab, char_vocab, tokenizer, max_word_len=5, max_seq_len=10
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for inputs, targets, char_inputs in dataloader:
        inputs, targets, char_inputs = inputs.to(device), targets.to(device), char_inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, char_inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Smoke Test Loss: {loss.item():.4f}")
        break

    print("Smoke test passed!")


def run_experiment(config_file, train_texts, val_texts, test_texts):

    print(f"Running experiment with config: {config_file}")
    config = load_config_from_csv(config_file)

    experiment_name = f"{config_file.split('.')[0]}"
    experiment_folder = f"results/{experiment_name}"
    os.makedirs(experiment_folder, exist_ok=True)

    csv_file_path = os.path.join(experiment_folder, f"{experiment_name}_results.csv")

    csv_headers = [
        "Epoch", "Loss (Cross-Entropy)", "Perplexity", "Accuracy", "Avg Levenshtein", "Execution Time (s)"
    ]
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

    print("Building vocabularies and datasets...")
    word_vocab, char_vocab, tokenizer = load_vocab_and_tokenizer(train_texts)

    train_dataset = TextDataset(
        train_texts, word_vocab, char_vocab, tokenizer,
        max_word_len=config["MAX_WORD_LEN"], max_seq_len=config["MAX_SEQ_LEN"]
    )
    val_dataset = TextDataset(
        val_texts, word_vocab, char_vocab, tokenizer,
        max_word_len=config["MAX_WORD_LEN"], max_seq_len=config["MAX_SEQ_LEN"]
    )
    test_dataset = TextDataset(
        test_texts, word_vocab, char_vocab, tokenizer,
        max_word_len=config["MAX_WORD_LEN"], max_seq_len=config["MAX_SEQ_LEN"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMWithCacheAndChar(
        word_vocab_size=len(word_vocab), char_vocab_size=len(char_vocab),
        word_embed_dim=config["WORD_EMBED_DIM"], char_embed_dim=config["CHAR_EMBED_DIM"],
        hidden_dim=config["HIDDEN_DIM"], char_hidden_dim=config["CHAR_HIDDEN_DIM"],
        num_layers=config["NUM_LAYERS"], cache_size=config["CACHE_SIZE"],
        dropout_rate=config["DROPOUT_RATE"], l2_lambda=config["L2_LAMBDA"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Training Phase
    train_model(
        model,
        train_loader,
        optimizer,
        device,
        config["EPOCHS"],
        csv_file_path,
        config["PATIENCE"],
        config["IMPROVEMENT_THRESHOLD"]
    )

    # Validation Phase
    evaluation_csv = os.path.join(experiment_folder, f"{experiment_name}_predictions.csv")
    summary_csv = os.path.join(experiment_folder, f"{experiment_name}_eval_summary.csv")
    evaluate_model(model, val_loader, device, word_vocab, summary_csv, evaluation_csv)

    # Test Phase
    summary_csv = os.path.join(experiment_folder, f"{experiment_name}_test_summary.csv")
    test_predictions_path = os.path.join(experiment_folder, f"{experiment_name}_test_predictions.csv")
    evaluate_model(model, test_loader, device, word_vocab, summary_csv, test_predictions_path)

    # Plot Training Results
    plot_training_results(csv_file_path, experiment_name)

    return csv_file_path, summary_csv


def main():
    experiments_dir = os.path.join(os.pardir, "experiments")

    smoke_test_enabled = os.getenv("SMOKE_TEST", "false").lower() == "true"
    if smoke_test_enabled:
        smoke_test()
        return

    experiment_configs = [
        os.path.join(experiments_dir, "1_baseline_config.csv"),
        os.path.join(experiments_dir, "2_red_params_fast_conv.csv"),
        os.path.join(experiments_dir, "3_larger_model.csv"),
        os.path.join(experiments_dir, "4_small_experiment.csv")
    ]

    # Load datasets
    print("Loading datasets...")
    train_texts, val_texts, test_texts = load_text_datasets()

    experiment_results = {}
    for config_file in experiment_configs:
        result_csv, summary_csv = run_experiment(config_file, train_texts, val_texts, test_texts)
        experiment_results[config_file.split('.')[0]] = (result_csv,summary_csv)

    # Plot aggregated results
    aggregated_results_folder = "results/final_evaluation"
    plot_aggregated_results(experiment_results, aggregated_results_folder)


if __name__ == "__main__":
    main()
