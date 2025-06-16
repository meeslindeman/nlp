import time
from pathlib import Path
import torch
from torch import nn, optim
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def prepare_example(batch, vocab, device):
    """
    Maps tokens to their IDs and prepares inputs and labels as tensors for a batch.

    Args:
        batch: A list of data examples, each with tokens and a label.
        vocab: Vocabulary object mapping tokens to IDs.
        device: Device to map the tensors to (CPU or GPU).

    Returns:
        x: Tensor of token IDs for the batch (padded).
        y: Tensor of labels for the batch.
    """
    # Extract token IDs and labels
    batch_tokens = [[vocab.w2i.get(t, 0) for t in example.tokens] for example in batch]
    batch_labels = [example.label for example in batch]

    # Determine the maximum sequence length in the batch
    max_len = max(len(tokens) for tokens in batch_tokens)

    # Pad all sequences to the maximum length
    padded_tokens = [tokens + [vocab.w2i.get("<pad>", 0)] * (max_len - len(tokens)) for tokens in batch_tokens]

    # Convert to tensors
    x = torch.LongTensor(padded_tokens).to(device)
    y = torch.LongTensor(batch_labels).to(device)
    return x, y

def simple_evaluate(model, data, prep_fn=prepare_example, device="cpu", average="macro", scores=False, **kwargs):
    """
    Evaluates model accuracy on a given dataset.
    
    Args:
        model: Trained model to evaluate.
        data: Dataset to evaluate on.
        prep_fn: Function to prepare input and label tensors.
        device: Device for tensor computations.

    Returns:
        correct: Number of correct predictions.
        total: Total number of predictions.
        accuracy: Accuracy as a float.
    """
    correct, total = 0, 0
    true_labels, predicted_labels = [], []
    model.eval()  # Disable dropout

    for example in data:
        # Wrap the single example in a list to make it compatible with prep_fn
        x, target = prep_fn([example], model.vocab, device)  # Note the list around `example`
        with torch.no_grad():
            logits = model(x)
        prediction = logits.argmax(dim=-1)

        correct += (prediction == target).sum().item()
        total += 1

        if scores: 
            true_labels.append(target.item())
            predicted_labels.append(prediction.item())
            precision = precision_score(true_labels, predicted_labels, average=average, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average=average, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
        
        else:
            precision = 0
            recall = 0
            f1 = 0

        accuracy = correct / float(total)

        metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return correct, total, metrics
    


def get_examples(data, shuffle=True, batch_size=1):
    """
    Yields batches of examples from the dataset, optionally shuffling them.

    Args:
        data: Dataset to sample from.
        shuffle: Boolean to shuffle the data before yielding.
        batch_size: Number of examples per batch.

    Yields:
        Batches of examples from the dataset.
    """
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        yield batch


def train_model(
    model, optimizer, train_data, dev_data, test_data,
    num_iterations=10000, print_every=1000, eval_every=1000,
    batch_fn=get_examples, prep_fn=prepare_example, eval_fn=simple_evaluate,
    batch_size=1, eval_batch_size=None, log_dir="logs", device="cpu", 
    scores=False, scheduler=None, patience=5
):
    """
    Trains a model with given parameters and tracks metrics for plotting.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    print(f"Training {model.__class__.__name__}...")

    criterion = nn.CrossEntropyLoss()
    best_eval, best_iter = 0., 0
    train_loss, iter_i = 0., 0
    no_improvement = 0
    stop_training = False
    losses, accuracies, steps = [], [], []
    start = time.time()

    if eval_batch_size is None:
        eval_batch_size = batch_size

    # Add tqdm progress bar
    with tqdm(total=num_iterations, desc="Training Progress", dynamic_ncols=True) as pbar:
        while iter_i < num_iterations and not stop_training:
            for batch in batch_fn(train_data, batch_size=batch_size):
                model.train()
                x, targets = prep_fn(batch, model.vocab, device)
                logits = model(x)
                loss = criterion(logits.view([targets.size(0), -1]), targets.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                iter_i += 1
                pbar.update(1)  # Update the progress bar

                if scheduler is not None:
                    scheduler.step()

                # Evaluate periodically
                if iter_i % eval_every == 0:
                    _, _, metrics = eval_fn(model, dev_data, prep_fn=prep_fn, device=device, scores=scores)
                    losses.append(train_loss / eval_every)  # Average loss over eval_every steps

                    accuracy = metrics["accuracy"]
                    accuracies.append(accuracy)
                    steps.append(iter_i)


                    # Reset train_loss after storing
                    train_loss = 0.

                    # Update the tqdm bar with custom metrics
                    pbar.set_postfix({
                        "Loss": f"{losses[-1]:.4f}",
                        "Accuracy": f"{accuracy:.4f}",
                        "LR": optimizer.param_groups[0]["lr"]
                    })

                    if accuracy > best_eval:
                        best_eval, best_iter = accuracy, iter_i
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_eval": best_eval,
                                "best_iter": best_iter,
                            },
                            log_dir / f"{model.__class__.__name__}.pt"
                        )
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    if no_improvement >= patience:
                        print(f"Early stopping triggered at iteration {iter_i}")
                        stop_training = True  # Set the flag to stop training
                        break

                if iter_i >= num_iterations:
                    stop_training = True  # Ensure stopping at the maximum iterations
                    break

    print("Done training")

    # Load the best model and evaluate
    print("Loading best model")
    path = log_dir / "{}.pt".format(model.__class__.__name__)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["state_dict"])

    _, _, train_metrics = eval_fn(
        model, train_data, batch_size=eval_batch_size,
        batch_fn=batch_fn, prep_fn=prep_fn)
    _, _, dev_metrics = eval_fn(
        model, dev_data, batch_size=eval_batch_size,
        batch_fn=batch_fn, prep_fn=prep_fn)
    _, _, test_metrics = eval_fn(
        model, test_data, batch_size=eval_batch_size,
        batch_fn=batch_fn, prep_fn=prep_fn)

    train_acc = train_metrics["accuracy"]
    dev_acc = dev_metrics["accuracy"]
    test_acc = test_metrics["accuracy"]

    print("Best model iter {:d}: "
          "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
              best_iter, train_acc, dev_acc, test_acc))

    metrics.update({
        "train_acc": train_acc,
        "dev_acc": dev_acc,
        "test_acc": test_acc
    })

    return losses, accuracies, metrics
