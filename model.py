import torch
import torch.nn as nn
from torchtyping import TensorType

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GPT(nn.Module):
    
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int, dropout: float):
        super().__init__()
        # torch.manual_seed(0)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)])
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
        # torch.manual_seed(0)
        tok_embed = self.token_embedding(context)
        pos_embed = self.position_embedding(torch.arange(context.shape[1]).to(device))
        embedded = tok_embed + pos_embed
        out = self.blocks(embedded)
        out = self.final_norm(out)
        return self.output_projection(out)


class TransformerBlock(nn.Module):
    
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        # torch.manual_seed(0)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # torch.manual_seed(0)
        x = self.norm1(embedded)
        x = self.attention(x) + embedded
        x = self.norm2(x)
        x = self.ff(x) + x
        return x


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        # torch.manual_seed(0)
        self.att_heads = nn.ModuleList([
            SingleHeadAttention(model_dim, model_dim // num_heads)
            for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # torch.manual_seed(0)
        heads = [head(embedded) for head in self.att_heads]
        concatenated = torch.cat(heads, dim=2)
        return self.dropout(self.output_proj(concatenated))


class SingleHeadAttention(nn.Module):

    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        # torch.manual_seed(0)
        self.key_gen = nn.Linear(model_dim, head_size, bias=False)
        self.query_gen = nn.Linear(model_dim, head_size, bias=False)
        self.value_gen = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # torch.manual_seed(0)
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        scores /= k.shape[2] ** 0.5

        mask = torch.tril(torch.ones(k.shape[1], k.shape[1])).to(device) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)

        return scores @ v


class FeedForward(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()
        # torch.manual_seed(0)
        self.up_proj = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_proj = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        # torch.manual_seed(0)
        return self.dropout(self.down_proj(self.relu(self.up_proj(x))))
