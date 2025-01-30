import os
import csv
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance

def train_model(model, dataloader, optimizer, device, epochs, csv_file_path, patience, improvement_threshold):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    best_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()
        total_loss = 0
        total_correct = 0
        total_valid = 0
        total_edit_distance = 0
        num_predictions = 0

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

            valid_mask = targets != criterion.ignore_index
            valid_outputs = outputs[valid_mask]
            valid_targets = targets[valid_mask]

            # Compute loss
            loss = criterion(valid_outputs, valid_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Compute predictions and accuracy
            predictions = torch.argmax(valid_outputs, dim=1)
            total_correct += (predictions == valid_targets).sum().item()
            total_valid += valid_targets.size(0)
            total_loss += loss.item()

            # Compute Levenshtein Distance
            idx_to_word = {idx: word for word, idx in dataloader.dataset.word_vocab.items()}
            for pred, tgt in zip(predictions, valid_targets):
                pred_word = idx_to_word.get(pred.item(), "<unk>")
                tgt_word = idx_to_word.get(tgt.item(), "<unk>")
                if tgt_word != "<pad>":
                    total_edit_distance += edit_distance(pred_word, tgt_word)
                    num_predictions += 1

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * total_correct / total_valid if total_valid > 0 else 0
        avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
        epoch_time = time.time() - epoch_start_time

        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_loss, perplexity, accuracy, avg_edit_distance, epoch_time])

        print(f"Epoch {epoch + 1} Summary: "
              f"Loss={avg_loss:.4f}, Perplexity={perplexity:.4f}, Accuracy={accuracy:.2f}%, "
              f"Levenshtein Distance={avg_edit_distance:.4f}, Time={epoch_time:.2f}s")

        # Early stopping logic
        if accuracy - best_accuracy > improvement_threshold:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


def evaluate_model(model, dataloader, device, vocab, output_file, num_samples_to_save=50):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    total_edit_distance = 0
    num_predictions = 0

    # Reverse the vocab dictionary for index-to-word mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}

    print("Starting Evaluation...")
    eval_start_time = time.time()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sample", "Input Sentence", "Target Word", "Predicted Word", "Levenshtein Distance"])

        saved_samples = 0

        with torch.no_grad():
            for inputs, targets, char_inputs in dataloader:
                inputs, targets, char_inputs = (
                    inputs.to(device),
                    targets.to(device),
                    char_inputs.to(device),
                )

                # Forward pass
                outputs = model(inputs, char_inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                # Filter valid tokens
                valid_mask = targets != criterion.ignore_index
                valid_outputs = outputs[valid_mask]
                valid_targets = targets[valid_mask]

                # Compute loss
                loss = criterion(valid_outputs, valid_targets)
                total_loss += loss.item()

                # Compute predictions and accuracy
                predictions = torch.argmax(valid_outputs, dim=1)
                total_correct += (predictions == valid_targets).sum().item()
                total_valid += valid_targets.size(0)

                # Save sample predictions
                if saved_samples < num_samples_to_save:
                    for pred, tgt in zip(predictions, valid_targets):
                        target_word = idx_to_word.get(tgt.item(), "<unk>")
                        predicted_word = idx_to_word.get(pred.item(), "<unk>")
                        lev_dist = edit_distance(predicted_word, target_word) if target_word != "<pad>" else "N/A"

                        writer.writerow([saved_samples + 1, "N/A", target_word, predicted_word, lev_dist])
                        saved_samples += 1

                        if target_word != "<pad>":
                            total_edit_distance += edit_distance(predicted_word, target_word)
                            num_predictions += 1

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float("inf")
    accuracy = 100.0 * total_correct / total_valid if total_valid > 0 else 0
    avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
    eval_time = time.time() - eval_start_time

    print(f"Evaluation Summary: "
          f"Loss={avg_loss:.4f}, "
          f"Perplexity={perplexity:.4f}, "
          f"Accuracy={accuracy:.2f}%, "
          f"Avg Levenshtein Distance={avg_edit_distance:.4f}, "
          f"Time={eval_time:.2f}s")

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
    os.makedirs(output_folder, exist_ok=True)

    all_data = []
    for experiment, csv_file in experiment_results.items():
        data = pd.read_csv(csv_file)
        data['Experiment'] = experiment
        all_data.append(data)
    all_data = pd.concat(all_data)

    metrics = ['Accuracy', 'Perplexity', 'Loss (Cross-Entropy)', 'Avg Levenshtein', 'Execution Time (s)']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for experiment in all_data['Experiment'].unique():
            exp_data = all_data[all_data['Experiment'] == experiment]
            plt.plot(exp_data['Epoch'], exp_data[metric], marker='o', label=experiment)
        plt.title(f'{metric} Across Experiments')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(output_folder, f"aggregated_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path)
        print(f"Saved aggregated plot to {plot_path}.")
