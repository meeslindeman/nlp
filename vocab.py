from collections import Counter, OrderedDict
from options import Options

# Initialize options
options = Options()

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen."""
    def __repr__(self):
        return f'{self.__class__.__name__}({OrderedDict(self)})'
    
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens."""
    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, token):
        """Counts the occurrence of a token."""
        self.freqs[token] += 1

    def add_token(self, token):
        """Adds a token to the vocabulary."""
        self.w2i[token] = len(self.w2i)
        self.i2w.append(token)

    def build(self, min_freq=0):
        """
        Builds the vocabulary by adding tokens with at least `min_freq` occurrences.

        Args:
            min_freq: Minimum number of occurrences for a word to be included.
        """
        self.add_token("<unk>")  # Reserve ID 0 for <unk> (unknown words)
        self.add_token("<pad>")  # Reserve ID 1 for <pad> (padding)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)

def create_vocabulary(train_data):
    """
    Creates and builds a vocabulary from the training dataset.

    Args:
        train_data: List of training examples, where each example has tokens.
        options: An Options instance with configuration settings.

    Returns:
        An instance of the Vocabulary class.
    """
    vocab = Vocabulary()
    for example in train_data:
        for token in example.tokens:
            vocab.count_token(token)
    vocab.build(min_freq=options.min_freq)
    return vocab

def sentiment_label_mappings():
    """
    Creates mappings for sentiment labels.

    Returns:
        i2t: List of labels mapped to their descriptions.
        t2i: Dictionary mapping descriptions to their corresponding numeric labels.
    """
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({label: idx for idx, label in enumerate(i2t)})
    return i2t, t2i
