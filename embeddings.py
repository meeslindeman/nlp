import numpy as np
import torch
from vocab import Vocabulary

def load_embeddings(path, embedding_dim):
    """
    Load the pre-trained embeddings from the file and create a new vocabulary.
    
    Args:
        path (str): Path to the embeddings file.
        embedding_dim (int): Dimension of the embeddings.
        
    Returns:
        vocab (Vocabulary): Vocabulary object with tokens from the embeddings.
        vectors (np.ndarray): Pre-trained embedding vectors.
    """
    vocab = Vocabulary()
    vectors = []

    # Add zero vector for <unk> and <pad>
    np.random.seed(42)
    unk = np.random.uniform(-0.05, 0.05, embedding_dim)
    pad = np.zeros(embedding_dim)

    vocab.add_token("<unk>")
    vocab.add_token("<pad>")

    vectors.append(unk)
    vectors.append(pad)

    # Load the embeddings from the file
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            word, *vector = line.split()
            vector = np.array(vector, dtype=np.float32)
            if len(vector) == embedding_dim:
                vocab.add_token(word)
                vectors.append(vector)

    vectors = np.stack(vectors)
    return vocab, vectors
