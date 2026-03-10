import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

DATA_PATH = "data/"


def download_data(url: str, file_name: str):
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, file_name)

    response = requests.get(url, timeout=30)

    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


def download_data_stream(url: str, file_name: str):
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, file_name)

    if Path(file_path).exists():
        print(f"{file_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True, timeout=60)

    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_tiny_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tiny_shakespeare.txt"
    download_data(url, filename)


def download_the_verdict_data():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    filename = "the_verdict.txt"
    download_data(url, filename)


def download_instruction_tuning_data():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    filename = "instruction_tuning_data.json"
    download_data(url, filename)


def read_file(file_name: str) -> str:
    full_path = os.path.join(DATA_PATH, file_name)

    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def read_tsv(file_name: str) -> pd.DataFrame:
    full_path = os.path.join(DATA_PATH, file_name)

    tsv = pd.read_csv(
        full_path,
        sep="\t",
    )

    return tsv


def read_json(file_name: str) -> List[Dict[str, str]]:
    full_path = os.path.join(DATA_PATH, file_name)

    with open(full_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    return json_data


def save_json(data: List[Dict[str, str]], file_name: str) -> None:
    full_path = os.path.join(DATA_PATH, file_name)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {file_name} to {full_path}")


def unzip_file(zip_name: str):
    full_path_zip = os.path.join(DATA_PATH, zip_name)

    with zipfile.ZipFile(full_path_zip, "r") as zip_ref:
        zip_ref.extractall(DATA_PATH)


def rename_tsv(file_name: str, new_file_name: str):
    full_path = os.path.join(DATA_PATH, file_name)

    os.rename(full_path, Path(DATA_PATH) / f"{new_file_name}.tsv")


def download_sms_spam_data():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    file_name = "sms_spam_collection.zip"

    download_data_stream(url, file_name)


def unzip_and_rename_sms_spam_data():
    zip_name = "sms_spam_collection.zip"
    unzip_file(zip_name)
    rename_tsv("SMSSpamCollection", "sms_spam")


# Qwen3 format
def format_instruction_tuning_data(entry: Dict[str, str]) -> Dict[str, str]:
    input_text = f": {entry['input']}" if entry["input"] else ""
    formatted_input = (
        f"<|im_start|>user\n{entry['instruction']}{input_text}\n<|im_end|>\n"
    )
    formatted_output = f"<|im_start|>assistant\n{entry['output']}\n<|im_end|>"

    return {
        "input": formatted_input,
        "output": formatted_output,
    }
