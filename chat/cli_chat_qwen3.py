# python -m chat.cli_chat_qwen3
# python -m chat.cli_chat_qwen3 --stream --cache --temperature 0.7 --top_k 30 --max_token_length 250 --model_type thinking
import argparse

import torch

from generate import Generator_Qwen_3
from models import Qwen_3_Model, get_qwen3_config
from utils import get_device

QWEN3_MODEL_CONFIG_0_6B = get_qwen3_config("0.6B")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 CLI Chat")

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable cache mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k for generation",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=250,
        help="Max token length for generation",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["thinking", "instruct", "base"],
        default="instruct",
        help="Model type",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()

    model = Qwen_3_Model(**QWEN3_MODEL_CONFIG_0_6B)

    # checkpoint = torch.load(
    #     "./checkpoints/checkpoint_tiny_shakespeare_final_ep11_step005889.pth",
    #     map_location=device,
    # )

    # model.load_state_dict(checkpoint["model_state_dict"])

    model.load_state_dict(
        torch.load(
            "./checkpoints/qwen3_0.6b_instruct.pth",
            map_location=device,
        )
    )
    model.to(device)

    generator = Generator_Qwen_3(
        model=model,
        num_layers=QWEN3_MODEL_CONFIG_0_6B["num_layers"],
        context_length=QWEN3_MODEL_CONFIG_0_6B["context_length"],
        model_size="0.6B",
        model_type=args.model_type,
        tokenizer_file_path="./tokenizer/qwen_3_instruct_tokenizer.json",
    )

    cache_enabled = args.cache
    temperature = args.temperature
    top_k = args.top_k
    max_token_length = args.max_token_length

    if args.stream:
        try:
            while True:
                user_input = input("You: ")

                if user_input.strip() == "exit_chat!":
                    print("Exiting chat...")
                    break

                input_ids = (
                    generator.text_to_token_ids(user_input).unsqueeze(0).to(device)
                )

                print("Qwen3: ", end="", flush=True)

                for token_id in generator.generate_stream(
                    idx=input_ids,
                    max_token_length=max_token_length,
                    temperature=temperature,
                    top_k=top_k,
                    cache_enabled=cache_enabled,
                ):
                    word = generator.token_ids_to_text(token_id.squeeze(0))
                    print(word, end="", flush=True)

                print("\n")
        except KeyboardInterrupt:
            print("\nBye!")
    else:
        try:
            while True:
                user_input = input("You: ")

                if user_input.strip() == "exit_chat!":
                    print("Exiting chat...")
                    break

                input_ids = (
                    generator.text_to_token_ids(user_input).unsqueeze(0).to(device)
                )

                print("Qwen3:")

                token_ids = generator.generate(
                    idx=input_ids,
                    max_token_length=max_token_length,
                    temperature=temperature,
                    top_k=top_k,
                    cache_enabled=cache_enabled,
                )
                generated_only_ids = token_ids[:, input_ids.shape[-1] :]
                response = generator.token_ids_to_text(generated_only_ids.squeeze(0))
                print(response)
        except KeyboardInterrupt:
            print("\nBye!")
