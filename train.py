import math
import gzip
import random
import tqdm
import numpy as np
from contextlib import nullcontext

import torch
from torch.optim import Adam
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize

from nGPT_pytorch import nGPT

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

USE_AMP = True
USE_PARAMETRIZE = True # whether to manually update weights after each optimizer step

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

assert not (USE_AMP and not torch.cuda.is_available())

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.5,
    min_p = 1e-1,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = min_p_filter(logits, min_p = min_p)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# nGPT char language model

model = nGPT(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    tied_embedding = True,
    add_value_residual = True,
    attn_norm_qk = False,
    manual_norm_weights = not USE_PARAMETRIZE
).to(device)

scaler = GradScaler(enabled = USE_AMP)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# if not using parametrize, register normalizing on optimizer step

if not USE_PARAMETRIZE:
    model.register_step_post_hook(optim)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        with torch.autocast(device_type = 'cuda',  dtype = torch.float16, enabled = USE_AMP):
            loss = model(data, return_loss = True)

        scaler.scale(loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")

    scaler.step(optim)
    scaler.update()

    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]

        prime = decode_tokens(inp)
        print(f"{prime} \n\n {'*' * 100}")

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\n\n{base_decode_output}\n")
