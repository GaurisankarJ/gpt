import torch
from models import GPT2_CONFIG_124M, GPT_2_Model
from utils.utils import generate_text_simple

if __name__ == "__main__":
    model = GPT_2_Model(GPT2_CONFIG_124M)

    x = generate_text_simple(model, torch.tensor([0, 0]).unsqueeze(0), 50, 256)
    print(x)
