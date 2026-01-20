# python -m chat.cli_chat
import tiktoken
import torch
from generate import Generator
from models import GPT_2_Model
from utils import count_parameters, get_device, text_to_token_ids

# GPT_2_MODEL_CONFIG = {
#     "vocab_size": 50257,  # Vocabulary size
#     "context_length": 128,  # Context length
#     "dim_embedding": 256,  # Embedding dimension
#     "num_heads": 4,  # Number of attention heads
#     "num_layers": 3,  # Number of layers
#     "drop_rate": 0.3,  # Dropout rate
#     "qkv_bias": False,  # Query-Key-Value bias
# }

GPT_2_MODEL_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "dim_embedding": 768,  # Embedding dimension
    "num_heads": 12,  # Number of attention heads
    "num_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias
}

if __name__ == "__main__":
    device = get_device()

    model = GPT_2_Model(GPT_2_MODEL_CONFIG_124M)
    print(f"{count_parameters(model):,} Parameters")

    # checkpoint = torch.load(
    #     "./checkpoints/checkpoint_tiny_shakespeare_final_ep11_step005889.pth",
    #     map_location=device,
    # )

    # model.load_state_dict(checkpoint["model_state_dict"])

    model.load_state_dict(
        torch.load(
            "./checkpoints/gpt2_small_124m.pth",
            map_location=device,
        )
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        context_length=GPT_2_MODEL_CONFIG_124M["context_length"],
    )

    try:
        while True:
            user_input = input("You: ")

            if user_input.strip() == "exit_chat!":
                print("Exiting chat...")
                break

            input_ids = text_to_token_ids(user_input).unsqueeze(0)

            print("GPT: ", end="", flush=True)

            for token_id in generator.generate(
                idx=input_ids,
                max_token_length=100,
                temperature=0.7,
                top_k=30,
                stream=True,
            ):
                word = tokenizer.decode([token_id.unsqueeze(0)])
                print(word, end="", flush=True)

            print("\n")
    except KeyboardInterrupt:
        print("\nBye!")
