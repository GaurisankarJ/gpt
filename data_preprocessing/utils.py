import os
import zipfile
from pathlib import Path

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
