from options import Options
from utils import set_seed, load_data, prepare_resources
from embeddings import load_embeddings
from batch_utils import get_minibatch, prepare_minibatch, evaluate, prepare_treelstm_minibatch
from vocab import create_vocabulary, sentiment_label_mappings
from models import BOW, CBOW, DeepCBOW, PTDeepCBOW, LSTMClassifier, TreeLSTMClassifier
from train import train_model
from torch import optim
import argparse
import torch
from pathlib import Path

# Initialize options
options = Options()

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models.")
    parser.add_argument("--model", type=str, default="BOW", help="Model type to train (BOW, CBOW, DeepCBOW)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)

    # Prepare all resources (dataset and embeddings)
    prepare_resources()

    # Load datasets
    datasets = load_data(lower=False)
    train_data = datasets["train"]
    dev_data = datasets["dev"]
    test_data = datasets["test"]

    glove_vocab, glove_vectors = load_embeddings("resources/glove.840B.300d.sst.txt", embedding_dim=300)
    word2vec_vocab, word2vec_vectors = load_embeddings("resources/googlenews.word2vec.300d.txt", embedding_dim=300)
    train_vocab = create_vocabulary(train_data)

    # Determine which vocab and vectors to use
    if args.model in ["PTDeepCBOW", "LSTM", "TreeLSTM"]:
        if options.pretrained_type == "glove":
            vocab, vectors = glove_vocab, glove_vectors
        elif options.pretrained_type == "word2vec":
            vocab, vectors = word2vec_vocab, word2vec_vectors
        else:
            raise ValueError(f"Unsupported pretrained type: {options.pretrained_type}")
    else:
        vocab, vectors = train_vocab, None

    # Create sentiment label mappings
    i2t, t2i = sentiment_label_mappings()

    # Retrieve the model based on the options
    if args.model == "BOW":
        model = BOW(vocab_size=len(vocab.w2i), embedding_dim=len(t2i), vocab=vocab)
    elif args.model == "CBOW":
        model = CBOW(
            vocab_size=len(vocab.w2i),
            embedding_dim=options.embedding_dim,
            output_dim=len(t2i),
            vocab=vocab
        )
    elif args.model == "DeepCBOW":
        model = DeepCBOW(
            vocab_size=len(vocab.w2i),
            embedding_dim=options.embedding_dim,
            hidden_dim=options.hidden_dim,
            output_dim=len(t2i),
            vocab=vocab
        )
    elif args.model == "PTDeepCBOW":
        model = PTDeepCBOW(
            vocab_size=len(vocab.w2i),
            embedding_dim=options.embedding_dim,
            hidden_dim=options.hidden_dim,
            output_dim=len(t2i),
            vocab=vocab
        )
        # Load pre-trained embeddings
        model.embed.weight.data.copy_(torch.from_numpy(vectors))
        model.embed.weight.requires_grad = False 
    elif args.model == "LSTM":
        model = LSTMClassifier(
            vocab_size=len(vocab.w2i),
            embedding_dim=options.embedding_dim,
            hidden_dim=options.hidden_dim,
            output_dim=len(t2i),
            vocab=vocab
        )
        if options.finetuning is False:
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(vectors))
                model.embed.weight.requires_grad = False
        elif options.finetuning is True:
            model.embed.weight.data.copy_(torch.from_numpy(vectors))
            model.embed.weight.requires_grad = True
    elif args.model == "TreeLSTM":
        model = TreeLSTMClassifier(
            vocab_size=len(vocab.w2i),
            embedding_dim=options.embedding_dim,
            hidden_dim=options.hidden_dim,
            output_dim=len(t2i),
            vocab=vocab
        )
        if options.finetuning is False:
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(vectors))
                model.embed.weight.requires_grad = False
        elif options.finetuning is True:
            model.embed.weight.data.copy_(torch.from_numpy(vectors))
            model.embed.weight.requires_grad = True
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Move model to the configured device
    model = model.to(options.device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

    if args.model == "LSTM" and options.mini_batch is True:
        # Train the model
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
            prep_fn=prepare_minibatch,
            eval_fn=evaluate,
            batch_size=options.batch_size,
            eval_batch_size=None,
            log_dir="logs",
            device=options.device,
            scores=options.scores
        )
    elif args.model == "TreeLSTM":
        # Train the model
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
            prep_fn=prepare_treelstm_minibatch,
            eval_fn=evaluate,
            batch_size=options.batch_size,
            eval_batch_size=None,
            log_dir="logs",
            device=options.device,
            scores=options.scores
        )
    else:
        # Train the model
        losses, accuracies, metrics = train_model(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            num_iterations=options.num_iterations,
            print_every=options.print_every,
            eval_every=options.eval_every,
            batch_size=options.batch_size,
            eval_batch_size=None,
            log_dir="logs",
            device=options.device,
            scores=options.scores
        )


    seed = torch.get_rng_state()
        
    # save losses and accuracies to files
    log_dir = Path(f"logs/{model.__class__.__name__}")
    if options.finetuning:
        log_dir = Path(f"logs/{model.__class__.__name__}_finetuning")
    log_dir.mkdir(exist_ok=True, parents=True)

    print(f"Final accuracy: {accuracies[-1]:.2f}")
    if options.scores:
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"F1 Score: {metrics['f1_score']:.2f}")

    with open(log_dir / f"loss_{seed[0].item()}.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    with open(log_dir / f"accuracy_{seed[0].item()}.txt", "w") as f:
        for acc in accuracies:
            f.write(f"{acc}\n")

if __name__ == "__main__":
    main()
