# import torch

# from data_preprocessing import (
#     InstructionDataLoader,
#     format_instruction_tuning_data,
#     read_json,
# )
from matplotlib import pyplot as plt
from pandas import read_csv

from models import get_qwen3_config

# from tokenizer import Qwen_3_Tokenizer

QWEN3_MODEL_CONFIG_0_6B = get_qwen3_config("0.6B")

# if __name__ == "__main__":
#     device = get_device()
#     print(f"Device: {device}")
#     model = Qwen_3_Model(**QWEN3_MODEL_CONFIG_0_6B)
#     model.load_state_dict(
#         torch.load(
#             "./checkpoints/qwen3_0.6b_instruct.pth",
#             map_location=device,
#         )
#     )
#     model.to(device)

#     tokenizer_base = Qwen_3_Tokenizer(
#         tokenizer_file_path="./tokenizer/qwen_3_instruct_tokenizer.json",
#         model_type="base",
#         apply_chat_template=False,
#         add_generation_prompt=False,
#         add_thinking=False,
#     )
#     tokenizer_instruct = Qwen_3_Tokenizer(
#         tokenizer_file_path="./tokenizer/qwen_3_instruct_tokenizer.json",
#         model_type="instruct",
#         apply_chat_template=True,
#         add_generation_prompt=True,
#         add_thinking=False,
#     )
#     tokenizer_thinking = Qwen_3_Tokenizer(
#         tokenizer_file_path="./tokenizer/qwen_3_instruct_tokenizer.json",
#         model_type="thinking",
#         apply_chat_template=True,
#         add_generation_prompt=True,
#         add_thinking=True,
#     )

#     # print(tokenizer_base.encode("Hello, world!"))
#     # print(tokenizer_base.decode(tokenizer_base.encode("Hello, world!")))

#     # print(tokenizer_instruct.encode("Hello, world!"))
#     # print(tokenizer_instruct.decode(tokenizer_instruct.encode("Hello, world!")))

#     # print(tokenizer_thinking.encode("Hello, world!"))
#     # print(tokenizer_thinking.decode(tokenizer_thinking.encode("Hello, world!")))

#     def generate(
#         tokenizer: Qwen_3_Tokenizer,
#         model: Qwen_3_Model,
#         idx: torch.Tensor,
#         device: torch.device,
#         max_token_length: int = 100,
#     ) -> torch.Tensor:
#         idx = idx.to(device)
#         model.eval()

#         for _ in range(max_token_length):
#             idx_cond = idx[:, -1024:]

#             logits = model(idx_cond)

#             logits = logits[:, -1, :]

#             idx_next = torch.argmax(logits, dim=-1, keepdims=True)

#             if tokenizer.eos_token_id is not None and torch.all(
#                 idx_next == tokenizer.eos_token_id
#             ):
#                 break

#             idx = torch.cat((idx, idx_next), dim=-1)

#         return idx

#     question = "What is the capital of France?"

#     idx = torch.tensor(tokenizer_base.encode(question)).unsqueeze(0)

#     print("########################################################")
#     print("Generating base...")
#     print("########################################################")
#     generated_ids_base = generate(
#         tokenizer=tokenizer_base,
#         model=model,
#         idx=idx,
#         device=device,
#         max_token_length=100,
#     )

#     print(tokenizer_base.decode(generated_ids_base.squeeze(0).tolist()))

#     idx = torch.tensor(tokenizer_instruct.encode(question)).unsqueeze(0)

#     print("########################################################")
#     print("Generating instruct...")
#     print("########################################################")
#     generated_ids_instruct = generate(
#         tokenizer=tokenizer_instruct,
#         model=model,
#         idx=idx,
#         device=device,
#         max_token_length=100,
#     )

#     print(tokenizer_instruct.decode(generated_ids_instruct.squeeze(0).tolist()))

#     idx = torch.tensor(tokenizer_thinking.encode(question)).unsqueeze(0)

#     print("########################################################")
#     print("Generating thinking...")
#     print("########################################################")
#     generated_ids_thinking = generate(
#         tokenizer=tokenizer_thinking,
#         model=model,
#         idx=idx,
#         device=device,
#         max_token_length=100,
#     )

#     print(tokenizer_thinking.decode(generated_ids_thinking.squeeze(0).tolist()))


if __name__ == "__main__":
    # device = get_device()
    # print(f"Device: {device}")
    # model = Qwen_3_Model(**QWEN3_MODEL_CONFIG_0_6B)
    # model.load_state_dict(
    #     torch.load(
    #         "./checkpoints/qwen3_0.6b_instruct.pth",
    #         map_location=device,
    #     )
    # )
    # model.to(device)

    # tokenizer_instruct = Qwen_3_Tokenizer(
    #     tokenizer_file_path="./tokenizer/qwen_3_instruct_tokenizer.json",
    #     model_type="instruct",
    #     apply_chat_template=True,
    #     add_generation_prompt=True,
    #     add_thinking=False,
    # )

    # data = read_json("instruction_tuning_data.json")

    # dataloader = InstructionDataLoader(
    #     tokenizer=tokenizer_instruct,
    #     batch_size=1,
    #     format_input=format_instruction_tuning_data,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0,
    #     ignore_index=70474,
    #     mask_inputs=True,
    # )

    # all_data = dataloader.create_dataloader(
    #     data=data,
    #     max_length=1024,
    #     device=torch.device("cpu"),
    # )
    # all_data = iter(all_data)
    # next_batch = next(all_data)
    # print(next_batch)
    # print("########################################################")
    # print("Input:")
    # print("########################################################")
    # print(tokenizer_instruct.decode(next_batch[0].squeeze(0).tolist()))
    # print("########################################################")
    # print("Target:")
    # print("########################################################")
    # print(tokenizer_instruct.decode(next_batch[1].squeeze(0).tolist()))

    # check_if_ollama_running()
    # result = query_model(prompt="What do Llamas eat?")
    # print(result)

    df = read_csv("logs/qwen3_0.6b_base_instruct_learning_rate_20260309_213431.csv")
    x = len(df)
    plt.ylabel("Learning Rate")
    plt.xlabel("Global Steps")
    plt.plot(range(x), df.iloc[:, 0].tolist())
    df.columns.values[0] = "Learning Rate"
    # df["Global Steps"] = range(x)
    print(df.head())
    df.to_csv(
        "logs/qwen3_0.6b_base_instruct_learning_rate_20260309_213431.csv", index=False
    )
    # df.rename(columns={"Global Steps": "x", "Learning Rate": "y"}, inplace=True)
    # df.head()
    # plt.show()
