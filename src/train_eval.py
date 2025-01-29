import os
import csv
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.metrics import edit_distance


def train_model(model,
                dataloader,
                optimizer,
                device,
                epochs,
                result_csv_file_path,
                patience,
                improvement_threshold):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    best_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        total_edit_distance = 0
        num_predictions = 0

        for inputs, targets, char_inputs, _ in dataloader:
            inputs, targets, char_inputs = (
                inputs.to(device),
                targets.to(device),
                char_inputs.to(device),
            )
            optimizer.zero_grad()

            outputs = model(inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()

            # Compute Levenshtein Distance
            idx_to_word = {idx: word for word, idx in dataloader.dataset.word_vocab.items()}
            for pred, tgt in zip(predictions, targets):
                pred_word = idx_to_word.get(pred.item(), "<unk>")
                tgt_word = idx_to_word.get(tgt.item(), "<unk>")
                if tgt_word != "<pad>":
                    total_edit_distance += edit_distance(pred_word, tgt_word)
                    num_predictions += 1

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * correct / total
        avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
        epoch_time = time.time() - epoch_start_time

        with open(result_csv_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_loss, perplexity, accuracy, avg_edit_distance, epoch_time])

        print(f"Epoch {epoch + 1} Summary: "
              f"Loss={avg_loss:.4f}, Perplexity={perplexity:.4f}, Accuracy={accuracy:.2f}%, "
              f"Levenshtein Distance={avg_edit_distance:.4f}, Time={epoch_time:.2f}s")

        if accuracy - best_accuracy > improvement_threshold:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


def evaluate_model(model,
                   dataloader,
                   device,
                   vocab,
                   output_file,
                   num_samples_to_save=50):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_edit_distance = 0
    num_predictions = 0

    # Reverse the vocab dictionary for index-to-word mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}

    print(f"Starting Evaluation for {output_file.split('csv')[0]}...")
    eval_start_time = time.time()

    # Open CSV file for logging
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Sample", "Input Sentence", "Target Word", "Predicted Word", "Levenshtein Distance", "Correct?"])

        saved_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack batch
                if len(batch) == 4:
                    inputs, targets, char_inputs, original_sentences = batch
                else:
                    raise ValueError("Dataloader must provide original sentences as the fourth item in the batch.")

                inputs, targets, char_inputs = (
                    inputs.to(device),
                    targets.to(device),
                    char_inputs.to(device),
                )

                # Forward pass
                outputs = model(inputs, char_inputs)
                batch_size, seq_len, _ = outputs.size()

                # Compute loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item()

                # Compute predictions
                predictions = torch.argmax(outputs, dim=-1)

                # Process each sequence in the batch
                for seq_idx in range(batch_size):
                    input_sentence = original_sentences[seq_idx]  # Retrieve the full sentence

                    for token_idx in range(seq_len):
                        target_token = targets[seq_idx, token_idx].item()
                        predicted_token = predictions[seq_idx, token_idx].item()

                        # Skip padding tokens (index 0)
                        if target_token == 0:
                            continue

                        target_word = idx_to_word.get(target_token, "<unk>")
                        predicted_word = idx_to_word.get(predicted_token, "<unk>")
                        lev_dist = edit_distance(predicted_word, target_word)

                        # Accuracy computation
                        is_correct = int(predicted_token == target_token)
                        correct += is_correct
                        total += 1

                        # Edit distance computation
                        total_edit_distance += lev_dist
                        num_predictions += 1

                        # Save sample to CSV
                        writer.writerow([
                            saved_samples + 1,
                            input_sentence,
                            target_word,
                            predicted_word,
                            lev_dist,
                            is_correct  # 1 if correct, 0 if incorrect
                        ])
                        saved_samples += 1

                        if saved_samples >= num_samples_to_save:
                            break

                    if saved_samples >= num_samples_to_save:
                        break

                if saved_samples >= num_samples_to_save:
                    break

    # Compute metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float("inf")
    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
    eval_time = time.time() - eval_start_time

    print(f"Evaluation Summary: "
          f"Loss={avg_loss:.4f}, "
          f"Perplexity={perplexity:.4f}, "
          f"Accuracy={accuracy:.2f}%, "
          f"Avg Levenshtein Distance={avg_edit_distance:.4f}, "
          f"Time={eval_time:.2f}s")

    with open(output_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])
        writer.writerow(["Evaluation Summary"])
        writer.writerow(["Loss (Cross-Entropy)", avg_loss])
        writer.writerow(["Perplexity", perplexity])
        writer.writerow(["Accuracy (%)", accuracy])
        writer.writerow(["Avg Levenshtein Distance", avg_edit_distance])
        writer.writerow(["Execution Time (s)", eval_time])

    return avg_loss, perplexity, accuracy, avg_edit_distance, eval_time

def plot_training_results(csv_file, experiment_name):
    os.makedirs(f"results/{experiment_name}/plots", exist_ok=True)

    # Read the results CSV
    data = pd.read_csv(csv_file)

    # Create a figure for plots
    plt.figure(figsize=(20, 10))

    # Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(data['Epoch'], data['Accuracy'], marker='o', label='Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # Perplexity
    plt.subplot(2, 3, 2)
    plt.plot(data['Epoch'], data['Perplexity'], marker='o', label='Perplexity', color='orange')
    plt.title('Perplexity Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()

    # Levenshtein Distance
    plt.subplot(2, 3, 3)
    plt.plot(data['Epoch'], data['Avg Levenshtein'], marker='o', label='Avg Levenshtein Distance', color='green')
    plt.title('Average Levenshtein Distance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Levenshtein Distance')
    plt.grid(True)
    plt.legend()

    # Cross-Entropy Loss
    plt.subplot(2, 3, 4)
    plt.plot(data['Epoch'], data['Loss (Cross-Entropy)'], marker='o', label='Cross-Entropy Loss', color='red')
    plt.title('Cross-Entropy Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(data['Epoch'], data['Execution Time (s)'], marker='o', label='Runtime per Epoch', color='purple')
    plt.title('Runtime Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = f"results/{experiment_name}/plots/{experiment_name}_training_results.png"
    plt.savefig(plot_path)
    print(f"Saved training plot to {plot_path}.")


def plot_aggregated_results(experiment_results, output_folder):
    """
    Plots and saves aggregated results for training, evaluation, and test data.

    Parameters:
    - experiment_results: dict, mapping experiment names to their respective CSV file paths.
    - output_folder: str, directory to save aggregated plots.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Define key metrics to track
    metrics = ["Accuracy", "Perplexity", "Loss (Cross-Entropy)", "Avg Levenshtein", "Execution Time (s)"]

    # Prepare a dataframe to store all results
    all_data = []

    # Load data from all experiments
    for exp_name, paths in experiment_results.items():
        for dataset, file_path in paths.items():
            if os.path.exists(file_path):  # Ensure file exists before reading
                df = pd.read_csv(file_path)
                df["Experiment"] = exp_name
                df["Dataset"] = dataset.capitalize()  # Store as 'Training', 'Evaluation', 'Test'
                all_data.append(df)

    if not all_data:
        print("No valid experiment data found. Skipping plotting.")
        return

    # Combine all data into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Generate plots for each metric
    for metric in metrics:
        if metric in all_data.columns:
            plt.figure(figsize=(12, 6))

            for experiment in all_data["Experiment"].unique():
                exp_data = all_data[all_data["Experiment"] == experiment]

                # Plot training, evaluation, and test results separately
                for dataset in ["Training", "Evaluation", "Test"]:
                    subset = exp_data[exp_data["Dataset"] == dataset]
                    if not subset.empty:
                        plt.plot(subset["Epoch"], subset[metric], marker='o', linestyle='-',
                                 label=f"{experiment} - {dataset}")

            plt.title(f'{metric} Across Experiments')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            # Save plot
            plot_filename = f"aggregated_{metric.lower().replace(' ', '_')}.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path)
            print(f"Saved aggregated plot: {plot_path}")
            plt.close()
        else:
            print(f"Skipping metric {metric} (not found in data).")