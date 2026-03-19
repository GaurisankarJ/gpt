import json
from typing import Any, Tuple

import psutil
import requests
import torch
import torch.nn as nn
from tqdm import tqdm


def check_if_running(process_name: str) -> bool:
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break

    return running


def check_if_ollama_running() -> None:
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

    print("Ollama running:", ollama_running)


def query_model(
    prompt: str,
    model: str = "llama3.2:3b",
    url="http://localhost:11434/api/chat",
) -> str:
    data: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {  # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }

    # Send the POST request
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


def generate_prompt(entry: dict[str, Any]) -> str:
    "Given the input `{format_input(entry)}` and correct output `{entry['output']}`, score the model response `{entry[json_key]}` on a scale from 0 to 100, where 100 is the best score. Respond with the integer number only."
    prompt = f"""
        Given the input `{entry["input"]}` and correct output `{entry["output"]}`, score the model response `{entry["model_response"]}` on a scale from 0 to 100, where 100 is the best score. 
        
        Respond with the integer number only.
        Return ONLY a number between 0 and 100.
        Do not write anything else.

        Input:
        {entry["input"]}

        Reference:
        {entry["output"]}

        Response:
        {entry["model_response"]}

        Score (0-100):
        """

    return prompt


def generate_model_scores(
    json_data: list[dict[str, Any]],
    model: str = "llama3.2:3b",
) -> list[int]:
    scores = []
    for entry in tqdm(json_data, desc="Scoring model responses"):
        prompt = generate_prompt(entry=entry)
        score = query_model(prompt=prompt, model=model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


@torch.inference_mode()
def calculate_next_token_probabilities(
    model: nn.Module,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if idx.ndim != 2:
        raise ValueError("idx must be a 2D tensor of shape [batch, seq_len].")
    if idx.shape[1] < 2:
        raise ValueError("idx must contain at least 2 tokens per sequence.")

    idx = idx.to(device)
    model.eval()

    # Autoregressive alignment: position t predicts token t+1.
    logits = model(idx)[:, :-1, :]
    probs = torch.softmax(logits, dim=-1)

    next_tok_idx = idx[:, 1:]
    next_tok_probs = probs.gather(
        dim=-1,
        index=next_tok_idx.unsqueeze(-1),
    ).squeeze(-1)

    next_tok_log_probs = torch.log(next_tok_probs.clamp_min(1e-12))
    joint_log_probs = next_tok_log_probs.sum(dim=-1)
    product_next_tok_probs = torch.exp(joint_log_probs)

    if show:
        print(f"Next token probabilities: {next_tok_probs}")
        print(f"Next token log probabilities: {next_tok_log_probs}")
        print(f"Product next token probabilities: {product_next_tok_probs}")
        print(f"Joint log probabilities: {joint_log_probs}")

    return (
        next_tok_probs,
        next_tok_log_probs,
        product_next_tok_probs,
        joint_log_probs,
    )


def calculate_average_joint_log_probability(
    model: nn.Module,
    prompt_idx: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> float:
    if prompt_idx.ndim == 0:
        raise ValueError("prompt_idx must contain at least one token.")

    prompt_length = prompt_idx.shape[-1]
    if prompt_length < 1:
        raise ValueError("prompt_idx must contain at least one token.")
    if idx.shape[-1] <= prompt_length:
        raise ValueError(
            "idx must contain at least one continuation token beyond prompt_idx."
        )

    _, next_tok_log_probs, _, _ = calculate_next_token_probabilities(
        model=model,
        idx=idx,
        device=device,
        show=show,
    )
    start_idx = prompt_length - 1
    return next_tok_log_probs[:, start_idx:].mean().item()


# Historical API compatibility
def calcualte_next_token_probabilities(
    model: nn.Module,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return calculate_next_token_probabilities(
        model=model,
        idx=idx,
        device=device,
        show=show,
    )


# Backward-compatible alias used by evaluation.__init__
def calculate_average_log_probability(
    model: nn.Module,
    prompt_idx: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    show: bool = False,
) -> float:
    return calculate_average_joint_log_probability(
        model=model,
        prompt_idx=prompt_idx,
        idx=idx,
        device=device,
        show=show,
    )
