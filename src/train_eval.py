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
        correct = 0
        total = 0

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

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * correct / total
        epoch_time = time.time() - epoch_start_time

        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_loss, perplexity, accuracy, epoch_time])

        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Perplexity={perplexity:.4f}, Accuracy={accuracy:.2f}%, Time={epoch_time:.2f}s")

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
    correct = 0
    total = 0
    total_edit_distance = 0
    num_predictions = 0

    # Reverse the vocab dictionary for index-to-word mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}

    print("Starting Evaluation...")
    eval_start_time = time.time()

    # Open CSV file for logging
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sample", "Input Sentence", "Target Word", "Predicted Word"])

        saved_samples = 0  # Counter for saved samples

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 4:  # Includes input sentences
                    inputs, targets, char_inputs, sentences = batch
                else:  # No sentences provided
                    inputs, targets, char_inputs = batch
                    sentences = [f"Sample {i+1}" for i in range(len(targets))]

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

                # Save only a subset of samples
                for i in range(len(targets)):
                    if saved_samples < num_samples_to_save:
                        input_sentence = sentences[i] if sentences else "N/A"
                        target_word = idx_to_word.get(targets[i].item(), "<unk>")
                        predicted_word = idx_to_word.get(predictions[i].item(), "<unk>")

                        writer.writerow([saved_samples + 1, input_sentence, target_word, predicted_word])
                        saved_samples += 1

                    # Ignore padding tokens for Levenshtein distance
                    if target_word != "<pad>":
                        total_edit_distance += edit_distance(predicted_word, target_word)
                        num_predictions += 1

                # Stop saving once the desired number of samples is reached
                if saved_samples >= num_samples_to_save:
                    break

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

    print(f"Evaluation Summary: "
          f"Loss={avg_loss:.4f}, "
          f"Perplexity={perplexity:.4f}, "
          f"Accuracy={accuracy:.2f}%, "
          f"Edit Distance={avg_edit_distance:.4f}, "
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
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Cross-Entropy Loss
    plt.subplot(2, 3, 4)
    plt.plot(data['Epoch'], data['Loss (Cross-Entropy)'], marker='o', label='Cross-Entropy Loss', color='red')
    plt.title('Cross-Entropy Loss Over Epochs')
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

    metrics = ['Accuracy', 'Perplexity', 'Loss (Cross-Entropy)', 'Execution Time (s)']
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
