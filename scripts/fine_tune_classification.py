# python -m scripts.fine_tune_classification --test True
import argparse

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import ClassificationDataLoader, read_tsv
from evaluation import Evaluator
from fine_tuning import TrainerClassification, classify_review
from models import GPT_2_Model
from utils import count_parameters, get_device

HYPERPARMETERS = {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.3,
    "num_epochs": 1,
    "freq_evaluation": 100,
    "iter_evaluation": 50,
}

GPT_2_MODEL_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "dim_embedding": 768,  # Embedding dimension
    "num_heads": 12,  # Number of attention heads
    "num_layers": 12,  # Number of layers
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias
}


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 Classification Script")

    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--test", action="store_true", help="Enable testing mode")

    return parser.parse_args()


def test_classifier(
    model: nn.Module,
    device: torch.device,
    test_dataloader: DataLoader,
):
    # Load GPT 2 fine-tuned to test
    checkpoint = torch.load(
        "./checkpoints/checkpoint_classification_fine_tuned_gpt2_final_classification_fine_tuned_ep01_step000260.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation
    evaluator = Evaluator(
        model=model,
        device=device,
    )

    test_accuracy = evaluator.calculate_accuracy_dataloader(
        dataloader=test_dataloader,
    )

    print(f"\n{'Dataset':<15} | {'Accuracy':<10}")
    print("-" * 28)
    print(f"{'Testing':<15} | {test_accuracy:>10.2%}")

    test_one = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    test_two = (
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    )

    test_texts = [test_one, test_two]

    print(f"\n{'CLASSIFICATION REPORT':^100}")
    print("-" * 100)

    for i, text in enumerate(test_texts, 1):
        result = classify_review(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=train_dataloader.dataset.max_length,
        )

        display_text = (text[:100] + "...") if len(text) > 100 else text

        print(f"Test {i}: {display_text}")
        print(f"Result: ** {result} **")
        print("-" * 100)


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    # Create GPT model
    model = GPT_2_Model(GPT_2_MODEL_CONFIG)
    print(f"{count_parameters(model):,} Parameters")

    # Load GPT 2
    model.load_state_dict(
        torch.load(
            "./checkpoints/gpt2_small_124m.pth",
            map_location=device,
        )
    )

    # Update model parameters
    for params in model.parameters():
        params.requires_grad = False
    num_classes = 2
    model.out_head = nn.Linear(
        in_features=GPT_2_MODEL_CONFIG["dim_embedding"],
        out_features=num_classes,
    )
    for params in model.transformer_blocks[-1].parameters():
        params.requires_grad = True
    for params in model.norm_final.parameters():
        params.requires_grad = True

    # Read tsv files to train on and test
    train_df = read_tsv("sms_spam_train.tsv")
    val_df = read_tsv("sms_spam_val.tsv")
    test_df = read_tsv("sms_spam_test.tsv")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Total training examples
    print(f"Total Examples: {len(train_df):,}")

    # Load text data
    dataloader = ClassificationDataLoader(
        tokenizer=tokenizer,
        context_length=GPT_2_MODEL_CONFIG["context_length"],
        batch_size=HYPERPARMETERS["batch_size"],
        pad_token="<|endoftext|>",
    )
    train_dataloader = dataloader.create_dataloader(
        dataframe=train_df,
    )
    val_dataloader = dataloader.create_dataloader(
        dataframe=val_df,
        max_length=train_dataloader.dataset.max_length,
    )
    test_dataloader = dataloader.create_dataloader(
        dataframe=test_df,
        max_length=train_dataloader.dataset.max_length,
    )

    # Evaluation
    evaluator = Evaluator(
        model=model,
        device=device,
    )
    train_accuracy = evaluator.calculate_accuracy_dataloader(
        dataloader=train_dataloader,
    )
    val_accuracy = evaluator.calculate_accuracy_dataloader(
        dataloader=val_dataloader,
    )
    test_accuracy = evaluator.calculate_accuracy_dataloader(
        dataloader=test_dataloader,
    )

    print(f"\n{'Dataset':<15} | {'Accuracy':<10}")
    print("-" * 28)
    print(f"{'Training':<15} | {train_accuracy:>10.2%}")
    print(f"{'Validation':<15} | {val_accuracy:>10.2%}")
    print(f"{'Testing':<15} | {test_accuracy:>10.2%}")

    if args.train:
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HYPERPARMETERS["learning_rate"],
            weight_decay=HYPERPARMETERS["weight_decay"],
        )

        # Create trainer
        trainer = TrainerClassification(
            model=model,
            optimizer=optimizer,
            device=device,
        )

        # Train
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=HYPERPARMETERS["num_epochs"],
            freq_evaluation=HYPERPARMETERS["freq_evaluation"],
            iter_evaluation=HYPERPARMETERS["iter_evaluation"],
            save_logs=True,
            model_name="gpt2",
        )

    if args.test:
        test_classifier(
            model=model,
            device=device,
            test_dataloader=test_dataloader,
        )
