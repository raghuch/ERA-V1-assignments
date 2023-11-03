from typing import List
from sklearn.base import validate_parameter_constraints
import torch

from tokenizer_utils import IntCharTokenizer
from conf import nanogpt_conf

BLOCK_SIZE = 256 #context length
BATCH_SIZE = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

def load_text() -> str:
    with open(nanogpt_conf["text_file"], "r") as f:
        text = f.read()
    return text

def load_int_char_tokenizer(text: str) -> IntCharTokenizer:
    return IntCharTokenizer(text)

def tokenize_char_to_int(text: str) -> List[int]:
    tokenizer = load_int_char_tokenizer(text)
    return tokenizer.encode(text)

# def decode_int_to_char(tokens: List[int]) -> str:
#     tokenizer = load_int_char_tokenizer(text)
#     return tokenizer.decode(tokens)

def load_text_as_tensor(text: str) -> torch.Tensor:
    data = torch.tensor(tokenize_char_to_int(text), dtype=torch.long)
    return data

def split_train_val(text):
    n = int(0.9 * len(text))
    train_data = text[:n]
    val_data = text[n:]

    return train_data, val_data


def get_random_batch(split):
    train_data, val_data = split_train_val(load_text_as_tensor(load_text()))
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


