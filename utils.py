import os
import re
import urllib.request  # Import for downloading files
import zipfile  # Import for extracting zip files
import torch
from pathlib import Path
from collections import namedtuple
from nltk import Tree

# Constants
SHIFT = 0
REDUCE = 1

# Basic functionality: file reader
def filereader(path):
    """Reads a text file and fixes an issue with '\\'."""
    path = "resources/" + path
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")

# Token extraction from sentiment trees
def tokens_from_treestring(s):
    """Extracts tokens from a sentiment tree."""
    return re.sub(r"\([0-9] |\)", "", s).split()

# Transition extraction for tree structure
def transitions_from_treestring(s):
    """Extracts transitions from a sentiment tree string."""
    s = re.sub(r"\([0-5] ([^)]+)\)", "0", s)
    s = re.sub(r"\)", " )", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\)", "1", s)
    return list(map(int, s.split()))

# Named tuple for examples
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

def examplereader(path, lower=False):
    """Reads examples from a file."""
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # Use NLTK's Tree
        label = int(line[1])
        transitions = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=transitions)

# Data loading function
def load_data(lower=False):
    """Loads train, dev, and test datasets."""
    datasets = {}
    for split in ["train", "dev", "test"]:
        path = f"trees/{split}.txt"
        datasets[split] = list(examplereader(path, lower=lower))
    return datasets

import random

# Function to shuffle tokens in an Example
def shuffle_tokens_in_example(example):
    shuffled_tokens = example.tokens[:]  # Copy the original tokens
    random.shuffle(shuffled_tokens)  # Shuffle the tokens in place
    return Example(tokens=shuffled_tokens, tree=example.tree, label=example.label, transitions=example.transitions)

# Function to shuffle tokens in the entire dataset
def shuffle_dataset(dataset):
    return [shuffle_tokens_in_example(example) for example in dataset]

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def download_file(url, output_path):
    """
    Download a file from a URL if it doesn't already exist.
    
    Args:
        url (str): URL of the file to download.
        output_path (str): Local path where the file will be saved.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}")
    else:
        print(f"{output_path} already exists. Skipping download.")

def prepare_resources(data_dir="resources"):
    """
    Ensure that all required datasets and embeddings are available.

    Args:
        data_dir (str): Directory to store datasets and embeddings.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset_url = "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
    dataset_zip = data_dir / "trainDevTestTrees_PTB.zip"
    dataset_folder = data_dir / "trees"

    if not dataset_folder.exists():
        download_file(dataset_url, dataset_zip)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted dataset to {dataset_folder}")
    else:
        print(f"Dataset already prepared in {dataset_folder}")

    # GloVe embeddings
    glove_url = "https://gist.githubusercontent.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt"
    glove_path = data_dir / "glove.840B.300d.sst.txt"
    download_file(glove_url, glove_path)

    # Word2Vec embeddings
    word2vec_url = "https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt"
    word2vec_path = data_dir / "googlenews.word2vec.300d.txt"
    download_file(word2vec_url, word2vec_path)

def extract_subtrees_from_tree(tree):
    """
    Extract subtrees from an NLTK Tree object.
    
    Args:
        tree: An NLTK Tree object.

    Returns:
        A list of subtrees as tuples (subtree, subtree string).
    """
    subtrees = []
    for i in range(3, 100):  # Height of the subtree to extract
        for t in tree.subtrees(lambda t: t.height() == i):
            subtrees.append((t, ' '.join(str(t).split())))
    return subtrees


def augment_with_subtrees(dataset, lower=False):
    """
    Augments a dataset with subtrees from each tree.
    
    Args:
        dataset: A list of Example objects, each containing a tree.
        lower: Whether to lowercase the tokens in the subtree strings.

    Returns:
        A list of augmented Example objects including subtrees.
    """
    augmented_data = []
    for example in dataset:
        tree = example.tree  # Assuming each example contains an NLTK Tree object
        subtrees = extract_subtrees_from_tree(tree)
        
        for subtree, subtree_string in subtrees:
            subtree_string = subtree_string.lower() if lower else subtree_string
            tokens = tokens_from_treestring(subtree_string)
            label = int(subtree_string[1])  # Extract root label
            transitions = transitions_from_treestring(subtree_string)
            augmented_data.append(Example(tokens=tokens, tree=subtree, label=label, transitions=transitions))
    return augmented_data



