# Sentiment Analysis with Neural Networks

This repository contains implementations of various neural network architectures for sentiment analysis, developed as part of an NLP course. The project explores different approaches to text classification, from simple bag-of-words models to more sophisticated tree-structured LSTMs.

## ⚠️ Academic Integrity Notice

**This code is shared for educational and portfolio purposes only.** If you are currently enrolled in a similar course:
- **DO NOT** copy or submit this code as your own work
- **DO NOT** use this as a template for your assignments
- Academic dishonesty policies apply - always do your own work
- Use this only as a reference to understand concepts and approaches

## Project Overview

This project implements and compares multiple neural network architectures for sentiment classification:

- **BOW (Bag of Words)**: Simple baseline using word frequency features
- **CBOW (Continuous Bag of Words)**: Averages word embeddings for classification
- **DeepCBOW**: Multi-layer extension of CBOW with hidden layers
- **PTDeepCBOW**: Pre-trained embedding version of DeepCBOW
- **LSTM**: Sequential model using Long Short-Term Memory networks
- **TreeLSTM**: Hierarchical model that processes syntactic tree structures

## Features

- **Multiple Model Architectures**: Six different approaches to sentiment analysis
- **Pre-trained Embeddings**: Support for GloVe and Word2Vec embeddings
- **Data Augmentation**: Optional dataset shuffling and subtree augmentation
- **Comprehensive Evaluation**: Accuracy, precision, recall, and F1-score metrics
- **Experiment Tracking**: Automatic logging of results and hyperparameters
- **Reproducible Results**: Seed-based random state management

## Requirements

This project uses conda for dependency management. Install the required packages using:

```bash
conda env create -f environment.yml
conda activate nlp
```

Alternatively, you can create the environment manually with the dependencies specified in `environment.yml`.

## Usage

### Basic Training

```bash
# Train a simple BOW model
python main.py --model BOW --seed 42

# Train LSTM with shuffled data
python main.py --model LSTM --seed 42 --shuffle

# Train TreeLSTM with node augmentation
python main.py --model TreeLSTM --seed 42 --node_augmentation
```

### Command Line Arguments

- `--model`: Model type (BOW, CBOW, DeepCBOW, PTDeepCBOW, LSTM, TreeLSTM)
- `--seed`: Random seed for reproducibility (default: 42)
- `--shuffle`: Shuffle word order in sentences
- `--node_augmentation`: Augment training data with syntactic subtrees

## Project Structure

```
├── main.py              # Main training script
├── options.py           # Configuration and hyperparameters
├── models/              # Model implementations
├── utils.py             # Data loading and preprocessing utilities
├── embeddings.py        # Pre-trained embedding handling
├── vocab.py             # Vocabulary creation and management
├── train.py             # Training loop implementation
├── batch_utils.py       # Batch processing and evaluation
├── logs/                # Training logs and results
└── resources/           # Pre-trained embeddings
    ├── glove.840B.300d.sst.txt
    └── googlenews.word2vec.300d.txt
```

## Model Details

### Architecture Comparison

| Model | Parameters | Complexity | Use Case |
|-------|------------|------------|----------|
| BOW | ~10K | Low | Baseline comparison |
| CBOW | ~100K | Low | Simple embedding averaging |
| DeepCBOW | ~300K | Medium | Non-linear transformations |
| PTDeepCBOW | ~300K | Medium | Leveraging pre-trained knowledge |
| LSTM | ~500K | High | Sequential processing |
| TreeLSTM | ~600K | Highest | Syntactic structure modeling |

### Learning Rates

The project uses model-specific learning rates optimized through experimentation:

- BOW: 0.005
- CBOW: 0.003
- DeepCBOW: 0.001
- PTDeepCBOW: 0.0005 (0.0001 when fine-tuning)
- LSTM: 0.0003
- TreeLSTM: 0.0002

## Results

Training results are automatically saved to CSV files with comprehensive metrics including:
- Model accuracy on train/dev/test sets
- Precision, recall, and F1-scores
- Hyperparameter configurations
- Training metadata

## Educational Context

This project was developed to explore fundamental concepts in neural NLP:
- Word embeddings and their properties
- Sequential vs. hierarchical text modeling
- Transfer learning in NLP applications
- Evaluation methodologies for text classification
- The evolution from traditional to neural approaches

This project is for educational purposes only. Please respect academic integrity policies if you're currently enrolled in coursework.

---

*Developed as part of an NLP course to explore various neural architectures for sentiment analysis.*