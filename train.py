from data import get_data, get_encoder_decoder

# from model_init import MY_MODEL as model, context_length, device
from model import GPT, device
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import shutil


file = "movies-quotes.txt"
sep = "~"
# file = "dad_jokes.csv"
# sep = ","

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

version = "v2"
batch_size = 64
max_iters = 7500
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
out_dir = "checkpoint"

# --------- TensorBoard Setup ---------
shutil.rmtree(f"runs/{file}-{version}", ignore_errors=True)

writer = SummaryWriter(log_dir=f"runs/{file}-{version}")
dummy_input = torch.zeros(1, context_length, dtype=torch.long).to(device)
writer.add_graph(model, dummy_input)


# --------------------------------------
# get a batch of data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# @torch.no_grad()
def estimate_loss():
    out = {}
    # model.eval()
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
    # model.train()
    return out


# @torch.no_grad()
def generate_test_samples(max_new_tokens=100):
    # model.eval()
    start_str = "A"
    context = torch.tensor([str_to_int.get(c, 0) for c in start_str], dtype=torch.long)[
        None, :
    ].to(device)
    for _ in range(max_new_tokens):
        if context.size(1) > context_length:
            context = context[:, -context_length:]
            # break
        logits = model(context)
        logits = logits[:, -1, :]  # last token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)
    result = "".join([int_to_str[int(i)] for i in context[0]])
    # model.train()
    return result


print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
LOG_MAX_LAYERS = 2
LOG_MAX_HEADS = 2

for iter in range(max_iters):
    xb, yb = get_batch("train")
    if iter % eval_interval == 0 or iter == max_iters - 1:
        with torch.no_grad():
            model.eval()
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # TensorBoard logging
            writer.add_scalar("Loss/train", losses["train"], iter)
            writer.add_scalar("Loss/val", losses["val"], iter)
            writer.add_scalar("Learning Rate", learning_rate, iter)
            writer.add_scalar("Iteration", iter, iter)
    
            generated_text = generate_test_samples(max_new_tokens=100)
            writer.add_text("Samples/Generated", generated_text, iter)
    
            _, attention_weights = model(xb[:1], return_attention=True)  # One sample
            for layer_idx, layer_attn in enumerate(attention_weights[:LOG_MAX_LAYERS]):
                # for head_idx, attn in enumerate(layer_attn[:LOG_MAX_HEADS]):
                #     attn_img = attn[0].unsqueeze(0)
                #     attn_img = attn_img / (attn_img.max() + 1e-9)  # normalize
                #     writer.add_image(
                #         f"Attention/layer{layer_idx}_head{head_idx}", attn_img, iter
                #     )
                attn_stack = torch.stack(
                    [attn[0] for attn in layer_attn[:LOG_MAX_HEADS]]
                )  # shape: [H, T, T]
                attn_stack = attn_stack / (attn_stack.max() + 1e-9)
                grid = vutils.make_grid(attn_stack.unsqueeze(1))  # [H,1,T,T] → grid
                writer.add_image(f"Attention/layer{layer_idx}_grid", grid, iter)

            for name, param in model.named_parameters():
                writer.add_histogram(f"Weights/{name}", param, iter)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, iter)

            if iter == 0:
                writer.add_embedding(
                    model.token_embedding.weight,
                    metadata=[int_to_str[i] for i in range(vocab_size)],
                    tag="Token Embeddings",
                    global_step=iter
                )            
            model.train()
            
    # forward pass
    logits = model(xb)

    # evaluate the model
    # logits = model(xb)
    # get loss
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = yb.view(B * T)
    loss = nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # for name, param in model.named_parameters():
    #     writer.add_histogram(f"Weights/{name}", param, iter)
    #     if param.grad is not None:
    #         writer.add_histogram(f"Gradients/{name}", param.grad, iter)

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
writer.close()

import test

test.test(os.path.join(out_dir, f"{file}-{version}.pt"))
