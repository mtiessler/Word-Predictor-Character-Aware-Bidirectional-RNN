import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from nltk.metrics import edit_distance

# TODO plot time for experiments
# TODO plot avg loss per epoch

def train_model(model, dataloader, optimizer, device, epochs, csv_file):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    results = []

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        total_loss = 0
        correct = 0
        total = 0
        total_edit_distance = 0
        num_predictions = 0

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

            # Debugging: Check inputs, targets, and outputs
            if torch.isnan(outputs).any():
                print("Found NaN in outputs!")
                return
            if targets.max() >= outputs.size(-1):
                print("Target index out of range!")
                print(f"Max target index: {targets.max()}, Output size: {outputs.size(-1)}")
                return

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
                if true != 0:  # Ignorar tokens de relleno
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
        results.append([epoch + 1, avg_loss, perplexity, accuracy, avg_edit_distance])

        print(f"Epoch {epoch + 1}/{epochs} Summary: "
              f"Avg Loss (Cross-Entropy): {avg_loss:.4f}, "
              f"Perplexity: {perplexity:.4f}, "
              f"Accuracy: {accuracy:.2f}%, "
              f"Avg Levenshtein: {avg_edit_distance:.4f}")

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    csv_file_path = os.path.join("results", csv_file)
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss (Cross-Entropy)", "Perplexity", "Accuracy", "Avg Levenshtein"])
        writer.writerows(results)

    print(f"Training Complete. Results saved to {csv_file_path}.")


def evaluate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_edit_distance = 0  # Para almacenar la distancia Levenshtein acumulada
    num_predictions = 0

    print("Starting Evaluation...")
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

            # Compute loss (Cross-Entropy)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Compute predictions
            predictions = torch.argmax(outputs, dim=1)

            # Calculate Levenshtein Distance
            for pred, true in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                if true != 0:  # Ignorar tokens de relleno
                    total_edit_distance += edit_distance(str(pred), str(true))
                    num_predictions += 1

            # Accuracy calculation
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)
            correct += batch_correct
            total += batch_total

    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)  # Entropía cruzada promedio
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = 100.0 * correct / total
    avg_edit_distance = total_edit_distance / num_predictions if num_predictions > 0 else 0

    print(f"Evaluation Complete: Avg Loss (Cross-Entropy): {avg_loss:.4f}, "
          f"Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.2f}%, "
          f"Avg Levenshtein Distance: {avg_edit_distance:.4f}")

    return (avg_loss,
            perplexity,
            accuracy,
            avg_edit_distance)


def plot_training_results(csv_file):
    os.makedirs("plots", exist_ok=True)
    data = pd.read_csv(os.path.join("results", csv_file))

    plt.figure(figsize=(18, 6))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(data['Epoch'], data['Accuracy'], marker='o', label='Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # Perplexity
    plt.subplot(1, 3, 2)
    plt.plot(data['Epoch'], data['Perplexity'], marker='o', label='Perplexity', color='orange')
    plt.title('Perplexity Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()

    # Levenshtein Distance
    plt.subplot(1, 3, 3)
    plt.plot(data['Epoch'], data['Avg Levenshtein'], marker='o', label='Avg Levenshtein Distance', color='green')
    plt.title('Average Levenshtein Distance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Levenshtein Distance')
    plt.grid(True)
    plt.legend()

    # Cross-Entropy Loss
    plt.subplot(1, 3, 3)
    plt.plot(data['Epoch'], data['Loss (Cross-Entropy)'], marker='o', label='Cross-Entropy Loss', color='red')
    plt.title('Cross-Entropy Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.grid(True)
    plt.legend()

    plot_file = os.path.join("plots", "training_results.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Training plot saved to {plot_file}.")
