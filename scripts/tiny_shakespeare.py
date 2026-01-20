import tiktoken
import torch
from data_preprocessing import BasicDataLoader, read_file
from generate import Generator
from models import GPT_2_Model
from pre_training import BasicTrainer
from utils import count_parameters, get_device

HYPER_PARMETERS = {
    "batch_size": 4,
    "learing_rate": 6.66e-5,
    "weight_decay": 0.3,
    "num_epochs": 10,
    "freq_evaluation": 100,
    "iter_evaluation": 50,
}

GPT_2_MODEL_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 128,  # Context length
    "dim_embedding": 256,  # Embedding dimension
    "num_heads": 16,  # Number of attention heads
    "num_layers": 3,  # Number of layers
    "drop_rate": 0.3,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}

if __name__ == "__main__":
    device = get_device()

    # Create GPT model
    model = GPT_2_Model(GPT_2_MODEL_CONFIG)
    print(f"{count_parameters(model):,} Parameters")

    # Read text file to train on
    text = read_file("tiny_shakespeare.txt")

    # Training/Validation split
    split = 0.90
    train_data = text[: int(split * len(text))]
    val_data = text[int(split * len(text)) :]

    tokenizer = tiktoken.get_encoding("gpt2")

    # Total training tokens
    print(f"Total Tokens: {len(tokenizer.encode(text)):,}")

    # Load text data
    dataloader = BasicDataLoader(
        tokenizer=tokenizer,
        context_length=GPT_2_MODEL_CONFIG["context_length"],
        stride=GPT_2_MODEL_CONFIG["context_length"],
        batch_size=HYPER_PARMETERS["batch_size"],
    )
    train_dataloader = dataloader.create_dataloader(train_data)
    val_dataloader = dataloader.create_dataloader(val_data)

    # Create optimizer
    optimzer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPER_PARMETERS["learing_rate"],
        weight_decay=HYPER_PARMETERS["weight_decay"],
    )

    # Create trainer
    trainer = BasicTrainer(
        model=model,
        optimizer=optimzer,
        device=device,
    )

    # Create Generator
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        context_length=128,
    )

    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=HYPER_PARMETERS["num_epochs"],
        freq_evaluation=HYPER_PARMETERS["freq_evaluation"],
        iter_evaluation=HYPER_PARMETERS["iter_evaluation"],
        save_logs=True,
        generator=generator,
        model_name="tiny_shakespeare",
    )
