from dataclasses import dataclass
import torch

@dataclass
class Options:
    seed: int = 42  
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  
    num_iterations: int = 30000  
    print_every: int = 1000 
    eval_every: int = 500 
    batch_size: int = 32  
    eval_batch_size: int = None  
    min_freq: int = 0

    embedding_dim: int = 300  
    hidden_dim: int = 168 # 168, 256 for LSTM
    pretrained_type: str = "word2vec"  # "glove" or "word2vec"

    scores: bool = True 
    mini_batch: bool = True
    finetuning: bool = False

    # Scheduler
    scheduler: bool = False
    step_size: int = 1000
    gamma: float = 0.1
    patience: int = 10  

    # TreeLSTM
    max_children: int = 2
    tree: str = "nary"  # "childsum" or "nary"
    
    

