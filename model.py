import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Union, Tuple, List

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context: TensorType[float]) -> TensorType[float]:
        tok_embed = self.token_embedding(context)
        pos_embed = self.position_embedding(torch.arange(context.shape[1], device=device))
        embedded = tok_embed + pos_embed
        out = self.blocks(embedded)
        out = self.final_norm(out)
        return self.output_projection(out)

    def get_attention_weights(self, context: TensorType[float]) -> List[TensorType[float]]:
        tok_embed = self.token_embedding(context)
        pos_embed = self.position_embedding(torch.arange(context.shape[1], device=device))
        embedded = tok_embed + pos_embed

        attention_weights = []
        x = embedded
        for block in self.blocks:
            attn = block.get_attention_weights(x)
            x = block.forward_only(x)
            attention_weights.append(attn)
        return attention_weights


class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        x = self.norm1(embedded)
        x = self.attention(x)
        x = x + embedded
        x = self.norm2(x)
        x = self.ff(x) + x
        return x

    def forward_only(self, embedded: TensorType[float]) -> TensorType[float]:
        return self.forward(embedded)

    def get_attention_weights(self, embedded: TensorType[float]) -> List[TensorType[float]]:
        x = self.norm1(embedded)
        return self.attention.get_attention_weights(x)


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.att_heads = nn.ModuleList(
            [
                SingleHeadAttention(model_dim, model_dim // num_heads)
                for _ in range(num_heads)
            ]
        )
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        heads = [head(embedded) for head in self.att_heads]
        concatenated = torch.cat(heads, dim=2)
        return self.dropout(self.output_proj(concatenated))

    def get_attention_weights(self, embedded: TensorType[float]) -> List[TensorType[float]]:
        all_weights = []
        for head in self.att_heads:
            weights = head.get_attention_weights(embedded)
            all_weights.append(weights)
        return all_weights


class SingleHeadAttention(nn.Module):

    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.key_gen = nn.Linear(model_dim, head_size, bias=False)
        self.query_gen = nn.Linear(model_dim, head_size, bias=False)
        self.value_gen = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        weights = self._calculate_attention(embedded)
        v = self.value_gen(embedded)
        return weights @ v

    def get_attention_weights(self, embedded: TensorType[float]) -> TensorType[float]:
        return self._calculate_attention(embedded)

    def _calculate_attention(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        scores /= k.shape[2] ** 0.5

        mask = torch.tril(torch.ones(k.shape[1], k.shape[1], device=embedded.device)) == 0
        scores = scores.masked_fill(mask, float("-inf"))
        return nn.functional.softmax(scores, dim=2)


class FeedForward(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_proj = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.dropout(self.down_proj(self.relu(self.up_proj(x))))
