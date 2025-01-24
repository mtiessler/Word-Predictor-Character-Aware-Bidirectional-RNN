import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.energy_meter import measure_energy
import time
from nltk.metrics import edit_distance


def train_model(model, dataloader, optimizer, device, epochs, csv_file_path):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    rapl_package = RaplPackageDomain(0)

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        total_edit_distance = 0
        num_predictions = 0

        with measure_energy(rapl_package) as energy_measurement:
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

                # Calculate Levenshtein Distance
                for pred, true in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                    if true != 0:  # Ignore padding tokens
                        total_edit_distance += edit_distance(str(pred), str(true))
                        num_predictions += 1

                correct += batch_correct
                total += batch_total
                total_loss += loss.item()

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = 100.0 * correct / total
        avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
        epoch_time = time.time() - epoch_start_time
        energy_consumed = energy_measurement[0].energy

        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                epoch + 1,
                avg_loss,
                perplexity,
                accuracy,
                avg_edit_distance,
                epoch_time,
                energy_consumed
            ])

        print(f"Epoch {epoch + 1}/{epochs} Summary: "
              f"Avg Loss (Cross-Entropy): {avg_loss:.4f}, "
              f"Perplexity: {perplexity:.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"Avg Levenshtein: {avg_edit_distance:.4f}, "
              f"Execution Time: {epoch_time:.2f}s, "
              f"Energy Consumption: {energy_consumed:.2f}J")

from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplPackageDomain
import time

def evaluate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_edit_distance = 0
    num_predictions = 0

    rapl_package = RaplPackageDomain(0)  # Energy measurement setup

    print("Starting Evaluation...")
    eval_start_time = time.time()
    with torch.no_grad(), measure_energy(rapl_package) as energy_measurement:
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

            # Compute loss (Cross-Entropy)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)

            # Calculate Levenshtein Distance
            for pred, true in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                if true != 0:  # Ignore padding tokens
                    total_edit_distance += edit_distance(str(pred), str(true))
                    num_predictions += 1

            # Accuracy calculation
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)
            correct += batch_correct
            total += batch_total

    # Metrics calculation
    avg_loss = total_loss / len(dataloader)  # Average cross-entropy loss
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = 100.0 * correct / total
    avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0
    eval_time = time.time() - eval_start_time
    energy_consumed = energy_measurement[0].energy

    print(f"Evaluation Complete:")
    print(f"  - Avg Loss (Cross-Entropy): {avg_loss:.4f}")
    print(f"  - Perplexity: {perplexity:.4f}")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - Avg Levenshtein Distance: {avg_edit_distance:.4f}")
    print(f"  - Execution Time: {eval_time:.2f}s")
    print(f"  - Energy Consumption: {energy_consumed:.2f}J")

    return avg_loss, perplexity, accuracy, avg_edit_distance, eval_time, energy_consumed

def plot_training_results(csv_file):
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Read the results CSV
    data = pd.read_csv(csv_file)

    # Create a figure for multiple plots
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

    # Average Levenshtein Distance
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

    # Execution Time
    plt.subplot(2, 3, 5)
    plt.plot(data['Epoch'], data['Execution Time (s)'], marker='o', label='Execution Time', color='purple')
    plt.title('Execution Time Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend()

    # Energy Consumption
    plt.subplot(2, 3, 6)
    plt.plot(data['Epoch'], data['Energy Consumption (J)'], marker='o', label='Energy Consumption', color='brown')
    plt.title('Energy Consumption Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Energy (J)')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_file = os.path.join("plots", "training_results.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Training plot saved to {plot_file}.")
