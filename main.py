import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from options import Options
from utils import set_seed, load_data, prepare_resources, shuffle_dataset, augment_with_subtrees
from embeddings import load_embeddings
from vocab import create_vocabulary, sentiment_label_mappings
from models import BOW, CBOW, DeepCBOW, PTDeepCBOW, LSTMClassifier, TreeLSTMClassifier
from train import train_model
from batch_utils import get_minibatch, prepare_minibatch, prepare_treelstm_minibatch, evaluate

# Initialize options globally
options = Options()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis models.")
    parser.add_argument("--model", type=str, default="BOW", help="Model type to train (BOW, CBOW, DeepCBOW, PTDeepCBOW, LSTM, TreeLSTM)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle word order in the dataset")
    parser.add_argument("--node_augmentation", action="store_true", default=False, help="Augment dataset with subtrees")
    return parser.parse_args()

def get_learning_rate(model_type, fine_tuning=False):
    """Returns the recommended initial learning rate based on the model type."""
    learning_rates = {
        "BOW": 0.005,
        "CBOW": 0.003,
        "DeepCBOW": 0.001,
        "PTDeepCBOW": 0.0005 if not fine_tuning else 0.0001,
        "LSTM": 0.0003,
        "TreeLSTM": 0.0002,
    }
    
    if model_type not in learning_rates:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return learning_rates[model_type]


def get_model(args, vocab, vectors, t2i):
    """Initialize the model based on the selected type and options."""
    model_map = {
        "BOW": BOW(len(vocab.w2i), len(t2i), vocab),
        "CBOW": CBOW(len(vocab.w2i), options.embedding_dim, len(t2i), vocab),
        "DeepCBOW": DeepCBOW(len(vocab.w2i), options.embedding_dim, options.hidden_dim, len(t2i), vocab),
        "PTDeepCBOW": PTDeepCBOW(len(vocab.w2i), options.embedding_dim, options.hidden_dim, len(t2i), vocab),
        "LSTM": LSTMClassifier(len(vocab.w2i), options.embedding_dim, options.hidden_dim, len(t2i), vocab),
        "TreeLSTM": TreeLSTMClassifier(len(vocab.w2i), options.embedding_dim, options.hidden_dim, len(t2i), vocab, tree=options.tree)
    }

    if args.model not in model_map:
        raise ValueError(f"Unknown model type: {args.model}")
    model = model_map[args.model]

    # Load pre-trained embeddings if applicable
    if args.model in ["PTDeepCBOW", "LSTM", "TreeLSTM"] and vectors is not None:
        with torch.no_grad():
            model.embed.weight.data.copy_(torch.from_numpy(vectors))
        model.embed.weight.requires_grad = options.finetuning

    return model


def train_selected_model(args, model, train_data, dev_data, test_data, scheduler=None):
    """Train the selected model."""
    prep_fn = prepare_minibatch if args.model != "TreeLSTM" else prepare_treelstm_minibatch

    if args.shuffle:
        train_data = shuffle_dataset(train_data)
        dev_data = shuffle_dataset(dev_data)
        test_data = shuffle_dataset(test_data)
        print("Shuffled dataset")
    
    if args.node_augmentation:
        train_data = augment_with_subtrees(train_data)
        print(f"Augmented dataset with subtrees: ({len(train_data)} training examples)")

    learning_rate = get_learning_rate(args.model, fine_tuning=options.finetuning)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses, accuracies, metrics = train_model(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        num_iterations=options.num_iterations,
        print_every=options.print_every,
        eval_every=options.eval_every,
        batch_fn=get_minibatch,
        prep_fn=prep_fn,
        eval_fn=evaluate,
        batch_size=options.batch_size,
        eval_batch_size=None,
        log_dir=f"logs/{model.__class__.__name__}/{args.seed}",
        device=options.device,
        scores=options.scores,
        scheduler=scheduler,
        patience=options.patience,
    )
    return losses, accuracies, metrics


def save_results(args, model, metrics, accuracies, losses):
    """Save training results to a file."""
    log_dir = Path(f"logs/{model.__class__.__name__}/{args.seed}")
    log_dir.mkdir(exist_ok=True, parents=True)

    learning_rate = get_learning_rate(args.model, fine_tuning=options.finetuning)

    results_file = Path(f"logs/{model.__class__.__name__}") / "results.csv"
    results_data = {
        "model": args.model,
        "seed": args.seed,
        "train_acc": metrics.get("train_acc"),
        "dev_acc": metrics.get("dev_acc"),
        "test_acc": metrics.get("test_acc"),
        "precision": metrics.get("precision", None),
        "recall": metrics.get("recall", None),
        "f1_score": metrics.get("f1_score", None),
        "num_iterations": options.num_iterations,
        "word_emb": options.pretrained_type,
        "hidden_dim": options.hidden_dim,
        "finetuned": options.finetuning,
        "lr": learning_rate,
        "tree": options.tree if args.model == "TreeLSTM" else "None",
        "shuffle": args.shuffle
    }

    # Update results file
    if results_file.exists():
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=results_data.keys())

    new_data = pd.DataFrame([results_data])
    results_df = pd.concat([results_df, new_data], ignore_index=True)
    results_df.to_csv(results_file, index=False)

    # Save accuracy and loss logs
    with open(log_dir / f"accuracy.txt", "w") as f:
        for acc in accuracies:
            f.write(f"{acc}\n")

    with open(log_dir / f"loss.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

def main():
    """Main script execution."""
    args = parse_arguments()
    set_seed(args.seed)
    prepare_resources()

    # Load datasets and embeddings
    datasets = load_data(lower=False)
    train_data, dev_data, test_data = datasets["train"], datasets["dev"], datasets["test"]

    glove_vocab, glove_vectors = load_embeddings("resources/glove.840B.300d.sst.txt", embedding_dim=300)
    word2vec_vocab, word2vec_vectors = load_embeddings("resources/googlenews.word2vec.300d.txt", embedding_dim=300)
    train_vocab = create_vocabulary(train_data)

    # Select vocab and vectors
    vocab, vectors = (train_vocab, None) if args.model not in ["PTDeepCBOW", "LSTM", "TreeLSTM"] else (
        glove_vocab if options.pretrained_type == "glove" else word2vec_vocab,
        glove_vectors if options.pretrained_type == "glove" else word2vec_vectors,
    )

    # Prepare model
    i2t, t2i = sentiment_label_mappings()
    model = get_model(args, vocab, vectors, t2i).to(options.device)

    # Initialize scheduler
    scheduler = StepLR(optim.Adam(model.parameters()), step_size=options.step_size, gamma=options.gamma) if options.scheduler else None

    # Train model
    losses, accuracies, metrics = train_selected_model(args, model, train_data, dev_data, test_data, scheduler)

    # Save results
    save_results(args, model, metrics, accuracies, losses)

    print(f"Final accuracy: {accuracies[-1]:.2f}")
    if options.scores:
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"F1 Score: {metrics['f1_score']:.2f}")


if __name__ == "__main__":
    main()
