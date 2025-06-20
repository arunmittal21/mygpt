from data import get_data, get_encoder_decoder

# from model_init import MY_MODEL as model, context_length, device
from model import GPT, device
import torch
from torch import nn
import os


# file = "movie-quotes.txt"
# sep = "~"
file = "dad_jokes.csv"
sep = ","

vocab_size, train_data, test_data, str_to_int, int_to_str = get_data(
    f"training_data/{file}", sep
)

# encode, decode = get_encoder_decoder(str_to_int, int_to_str)

context_length = 128  # block_size
model_dim = 252
num_blocks = 6
num_heads = 6
dropout = 0.2
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads, dropout).to(
    device
)

version = "v1"
batch_size = 64
max_iters = 100
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
out_dir = "checkpoint"


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = Y.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    # evaluate the model
    logits = model(xb)
    # get loss
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = yb.view(B * T)
    loss = nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# context_length = 128 #block_size
# model_dim = 252
# num_blocks = 6
# num_heads = 6
# dropout = 0.2
# MY_MODEL = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads, dropout).to(device)

checkpoint = {
    "model": model.state_dict(),
    "vocab_size": vocab_size,
    "context_length": context_length,
    "model_dim": model_dim,
    "num_blocks": num_blocks,
    "num_heads": num_heads,
    "dropout": 0.2,
    "str_to_int": str_to_int,
    "int_to_str": int_to_str,
}

print(f'saving checkpoint to {os.path.join(out_dir, f"{file}-{version}.pt")}')
torch.save(checkpoint, os.path.join(out_dir, f"{file}-{version}.pt"))

import test

test.test(os.path.join(out_dir, f"{file}-{version}.pt"))
