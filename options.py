from dataclasses import dataclass
import torch

@dataclass
class Options:
    seed: int = 100  # Seed for reproducibility
    min_freq: int = 0  # Minimum frequency for vocabulary inclusion
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use
    learning_rate: float = 0.0005  # Learning rate for BOW models;  lr=3e-4 for LSTM or 2e-4
    num_iterations: int = 40000  # Number of training iterations
    print_every: int = 1000  # Interval for printing training logs
    eval_every: int = 1000  # Interval for evaluation
    batch_size: int = 25  # Batch size for training
    eval_batch_size: int = None  # Batch size for evaluation (default: same as batch_size)
    embedding_dim: int = 300  
    hidden_dim: int = 168    
    pretrained_type: str = "glove"  # Pre-trained embeddings type: "glove" or "word2vec"
    scores: bool = False # Whether to compute precision, recall, and F1 scores
    mini_batch: bool = False # Whether to use mini-batch training in LSTM
    finetuning: bool = True # Whether to fine-tune pre-trained embeddings

