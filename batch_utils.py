import torch
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def get_minibatch(data, batch_size=25, shuffle=True):
  """Return minibatches, optional shuffling"""

  if shuffle:
    # print("Shuffling training data")
    random.shuffle(data)  # shuffle training data each epoch

  batch = []

  # yield minibatches
  for example in data:
    batch.append(example)

    if len(batch) == batch_size:
      yield batch
      batch = []

  # in case there is something left
  if len(batch) > 0:
    yield batch

def pad(tokens, length, pad_value=1):
  """add padding 1s to a sequence to that it has the desired length"""
  return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab, device):
  """
  Minibatch is a list of examples.
  This function converts words to IDs and returns
  torch tensors to be used as input/targets.
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])

  # vocab returns 0 if the word is not there
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

  x = torch.LongTensor(x)
  x = x.to(device)

  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  return x, y

def evaluate(
    model, data,
    batch_fn=get_minibatch, prep_fn=prepare_minibatch,
    batch_size=16, device="cpu", scores=False, average="macro"
):
    """
    Evaluates model performance on a given dataset using mini-batches.
    
    Args:
        model: Trained model to evaluate.
        data: Dataset to evaluate on.
        batch_fn: Function to generate mini-batches.
        prep_fn: Function to prepare mini-batches.
        batch_size: Number of examples per mini-batch.
        device: Device for tensor computations ("cpu" or "cuda").
        scores: Whether to compute precision, recall, and F1 scores.
        average: Type of averaging for precision, recall, and F1 
                 ("micro", "macro", "weighted"). Default is "macro".
    
    Returns:
        correct: Total number of correct predictions.
        total: Total number of examples.
        metrics: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    correct = 0
    total = 0
    true_labels, predicted_labels = [], []
    model.eval()  # Disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab, device)
        with torch.no_grad():
            logits = model(x)

        predictions = logits.argmax(dim=-1).view(-1)

        # Collect true and predicted labels if scores are required
        if scores:
            true_labels.extend(targets.view(-1).tolist())
            predicted_labels.extend(predictions.tolist())

        # Update accuracy
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    # Compute additional metrics if required
    if scores:
        precision = precision_score(true_labels, predicted_labels, average=average, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average=average, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
    else:
        precision = recall = f1 = 0

    # Compute accuracy
    accuracy = correct / float(total)

    # Return metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return correct, total, metrics

# Tree LSTM

# Helper functions for batching and unbatching states
# For speed we want to combine computations by batching, but
# for processing logic we want to turn the output into lists again
# to easily manipulate.

def batch(states):
  """
  Turns a list of states into a single tensor for fast processing.
  This function also chunks (splits) each state into a (h, c) pair"""
  return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
  """
  Turns a tensor back into a list of states.
  First, (h, c) are merged into a single state.
  Then the result is split into a list of sentences.
  """
  return torch.split(torch.cat(state, 1), 1, 0)

def prepare_treelstm_minibatch(mb, vocab, device):
  """
  Returns sentences reversed (last word first)
  Returns transitions together with the sentences.
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])

  # vocab returns 0 if the word is not there
  # NOTE: reversed sequence!
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

  x = torch.LongTensor(x)
  x = x.to(device)

  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  maxlen_t = max([len(ex.transitions) for ex in mb])
  transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
  transitions = np.array(transitions)
  transitions = transitions.T  # time-major

  return (x, transitions), y