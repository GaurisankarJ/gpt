import os

import requests

DATA_PATH = "data/"


def download_data(url: str, filename: str):
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, filename)

    response = requests.get(url, timeout=30)

    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


def download_tiny_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tiny_shakespeare.txt"
    download_data(url, filename)


def download_the_verdict_data():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    filename = "the_verdict.txt"
    download_data(url, filename)


def read_file(file_name: str) -> str:
    full_path = os.path.join(DATA_PATH, file_name)

    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text
