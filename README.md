
# LSTM Recurrent Neural Language Model (RNLM) with Character-Aware Embeddings and Caching

This project implements an LSTM-based Recurrent Neural Language Model (RNLM) for next-word prediction. The model incorporates **character-aware embeddings**, **caching mechanisms**, and optional **sampling-based approximations** to improve efficiency and reduce perplexity (PPL). The project is modularized for flexibility and can be executed in a Dockerized environment with configurable options via command-line arguments.

## **Main Purpose**
The primary goal of this project is to demonstrate advanced techniques in neural language modeling:
- **Character-Aware Models**: Enhances word representations by incorporating character-level features.
- **Caching**: Leverages previously computed results to improve long-term dependency modeling.
- **Sampling-Based Approximations**: Reduces computational overhead for large vocabularies during training.

The implementation is designed to be:
1. **Modular**: Each component (dataset processing, model definition, training) is encapsulated in a separate file.
2. **Flexible**: Features like caching and sampling can be toggled using command-line arguments.
3. **Scalable**: Suitable for both small (WikiText-2) and large (20 Newsgroups) corpora.

---

## **Files Overview**

### 1. `dataset.py`
This file handles the preprocessing and loading of datasets.

- **Functions**:
  - `clean_text(text)`: Cleans input text by removing unwanted characters and converting it to lowercase.
  - `tokenize_by_char(text)`: Tokenizes text into individual characters.
  - `load_vocab_and_tokenizer()`: Generates word and character vocabularies using the WikiText-2 dataset.
  - `TextDataset`: A PyTorch `Dataset` class that prepares word-level and character-level sequences for the model.
  - `collate_fn(batch)`: Custom collate function for padding variable-length sequences to the same length in each batch.
  - `get_dataloaders(...)`: Creates PyTorch `DataLoader` objects for training and validation data.

- **Purpose**: Provides a robust pipeline for data preparation, including tokenization, vocabulary building, and batching.

---

### 2. `model.py`
This file defines the LSTM-based model with support for character-aware embeddings and caching.

- **Class: `LSTMWithCacheAndChar`**
  - **Features**:
    - **Word Embeddings**: Maps words to dense vector representations.
    - **Character Embeddings**: Extracts character-level features using an LSTM.
    - **Caching**: Stores previously computed logits for reuse, improving computational efficiency.
  - **Components**:
    - `__init__`: Initializes embeddings, LSTM layers, and caching parameters.
    - `forward`: Processes input sequences to produce logits and updates the cache.

- **Purpose**: Encapsulates the architecture and forward computation of the RNLM.

---

### 3. `train.py`
This file handles the training and evaluation of the model.

- **Functions**:
  - `train_model(...)`: Implements the training loop with support for standard cross-entropy loss.
  - `evaluate_model(...)`: Evaluates the model on a validation set and computes perplexity (PPL).

- **Main Script**:
  - Parses command-line arguments to configure features like caching, sampling, batch size, learning rate, and epochs.
  - Loads datasets and initializes the model.
  - Trains the model using the specified configuration.
  - Evaluates the trained model and outputs metrics like perplexity.

- **Purpose**: Provides a complete pipeline for training and validating the LSTM-RNLM.

---

### 4. `requirements.txt`
Lists the Python dependencies required for the project.

- **Contents**:
  - `torch`: PyTorch library for building and training neural networks.
  - `torchtext`: For handling text datasets and tokenization.

---

### 5. `Dockerfile`
Defines a Docker container for running the project in a consistent environment.

- **Steps**:
  1. Starts from the official Python 3.9 slim image.
  2. Installs dependencies from `requirements.txt`.
  3. Copies project files into the container.
  4. Sets the default command to execute `train.py`.

- **Purpose**: Ensures portability and reproducibility of the training and evaluation pipeline.

---
## Datasets
- 20 newsgroups: http://qwone.com/~jason/20Newsgroups/
- Reuters: http://www.daviddlewis.com/resources/testcollections/reuters21578/
- Fake News Dataset: https://github.com/GeorgeMcIntire/fake_real_news_dataset
---

## **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_name>
```

### **2. Build the docker image**
```bash
docker build -t lstm-rnlm .
```

```bash
docker run --rm lstm-rnlm --use_caching --batch_size 32 --epochs 10 --lr 0.001
```

### **3. Run Locally (Optional)**
```bash
pip install -r requirements.txt
```

```bash
python train.py --use_caching --batch_size 32 --epochs 10 --lr 0.001
```

---
## Command-Line Arguments

| Argument        | Description                                              | Default Value |
|-----------------|----------------------------------------------------------|---------------|
| `--use_caching` | Enables the caching mechanism during training.           | `False`       |
| `--batch_size`  | Sets the batch size for training and evaluation.          | `32`          |
| `--epochs`      | Specifies the number of training epochs.                 | `10`          |
| `--lr`          | Configures the learning rate for the optimizer.          | `0.001`       |

---

## Key Features

### **Character-Aware Embeddings**
- Improves handling of rare and out-of-vocabulary words by learning subword patterns.

### **Caching**
- Reduces redundant computations by reusing results from previous forward passes.

### **Sampling-Based Approximation (Optional)**
- Speeds up training for large vocabularies by approximating the softmax computation.

### **Dockerized Environment**
- Ensures consistency and reproducibility across different systems.

---

## Future Enhancements

1. **Integration of Attention Mechanisms**: Improve the model's ability to focus on relevant context.
2. **Support for Custom Datasets**: Extend preprocessing and training to handle user-provided datasets.
3. **Dynamic Hyperparameter Tuning**: Automate the optimization of parameters like learning rate and embedding size.
