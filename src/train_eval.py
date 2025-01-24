import os
import csv
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.energy_meter import measure_energy, EnergyMeter
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
        correct = 0
        total = 0

        for inputs, targets, char_inputs in dataloader:
            inputs, targets, char_inputs = (
                inputs.to(device),
                targets.to(device),
                char_inputs.to(device),
            )
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, char_inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)

            correct += batch_correct
            total += batch_total
            total_loss += loss.item()

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * correct / total
        epoch_time = time.time() - epoch_start_time

        # Log epoch metrics
        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                epoch + 1,
                avg_loss,
                perplexity,
                accuracy,
                epoch_time,
            ])

        print(f"Epoch {epoch + 1}/{epochs} Summary: "
              f"Avg Loss (Cross-Entropy): {avg_loss:.4f}, "
              f"Perplexity: {perplexity:.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"Execution Time: {epoch_time:.2f}s")

        # Early stopping logic
        if accuracy - best_accuracy > improvement_threshold:
            best_accuracy = accuracy
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. "
                  f"No significant improvement in accuracy for {patience} consecutive epochs.")
            break


def evaluate_model(model, dataloader, device, vocab, output_file):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_edit_distance = 0
    num_predictions = 0

    # Reverse the vocab dictionary to get index-to-word mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}

    print("Starting Evaluation...")
    eval_start_time = time.time()

    # Open CSV file for logging
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sentence", "Target Word", "Predicted Word"])

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 4:  # Includes sentences
                    inputs, targets, char_inputs, sentences = batch
                else:  # No sentences
                    inputs, targets, char_inputs = batch
                    sentences = None  # Placeholder

                inputs, targets, char_inputs = (
                    inputs.to(device),
                    targets.to(device),
                    char_inputs.to(device),
                )

                # Forward pass
                outputs = model(inputs, char_inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                # Compute loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Compute predictions
                predictions = torch.argmax(outputs, dim=1)

                # Write predictions to CSV
                for i in range(len(targets)):
                    target_word = idx_to_word.get(targets[i].item(), "<unk>")
                    predicted_word = idx_to_word.get(predictions[i].item(), "<unk>")
                    sentence = sentences[i] if sentences else f"Sample {i + 1}"
                    writer.writerow([sentence, target_word, predicted_word])

                    # Ignore padding tokens for Levenshtein distance
                    if target_word != "<pad>":
                        total_edit_distance += edit_distance(predicted_word, target_word)
                        num_predictions += 1

                # Accuracy calculation
                batch_correct = (predictions == targets).sum().item()
                batch_total = targets.size(0)
                correct += batch_correct
                total += batch_total

    # Metrics calculation
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = 100.0 * correct / total
    avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
    eval_time = time.time() - eval_start_time

    print(f"Evaluation Complete: "
          f"Avg Loss: {avg_loss:.4f}, "
          f"Perplexity: {perplexity:.4f}, "
          f"Accuracy: {accuracy:.2f}%, "
          f"Avg Levenshtein Distance: {avg_edit_distance:.4f}, "
          f"Time: {eval_time:.2f}s")

    return avg_loss, perplexity, accuracy, avg_edit_distance, eval_time


def plot_training_results(csv_file, experiment_name):
    os.makedirs("plots", exist_ok=True)

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
    plt.ylabel('Avg Levenshtein Distance')
    plt.grid(True)
    plt.legend()

    # Cross-Entropy Loss
    plt.subplot(2, 3, 4)
    plt.plot(data['Epoch'], data['Loss (Cross-Entropy)'], marker='o', label='Cross-Entropy Loss', color='red')
    plt.title('Cross-Entropy Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.grid(True)
    plt.legend()

    if 'Execution Time (s)' in data.columns:
        plt.subplot(2, 3, 5)
        plt.plot(data['Epoch'], data['Execution Time (s)'], marker='o', label='Execution Time', color='purple')
        plt.title('Execution Time Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.grid(True)
        plt.legend()
    else:
        print("Warning: 'Execution Time (s)' column is missing in the CSV file.")

    # Save the plot
    plot_file = os.path.join("plots", f"{experiment_name}_training_results.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Training plot saved to {plot_file}.")
