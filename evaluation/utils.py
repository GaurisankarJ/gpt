import json
from typing import Any

import psutil
import requests
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
