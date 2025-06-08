import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GPT(nn.Module):
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    self.key_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.query_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.value_layer = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.key_layer(embedded)
                    q = self.query_layer(embedded)
                    v = self.value_layer(embedded)

                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = (lower_triangular == 0).to(device)
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v

            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                self.attention_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.attention_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.compute = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, embedded):
                head_outputs = []
                for head in self.attention_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.dropout(self.compute(concatenated))

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2

            def forward(self, x):
                return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.vanilla_nn = self.VanillaNeuralNetwork(model_dim)
            self.layer_norm_one = nn.LayerNorm(model_dim)
            self.layer_norm_two = nn.LayerNorm(model_dim)

        def forward(self, embedded):
            embedded = embedded + self.mhsa(self.layer_norm_one(embedded)) # skip connection
            embedded = embedded + self.vanilla_nn(self.layer_norm_two(embedded)) # another skip connection
            return embedded

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.layer_norm_three = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        embedded = self.token_embedding(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length).to(device)
        embedded = embedded + self.pos_embedding(positions)

        raw_output = self.vocab_projection(self.layer_norm_three(self.transformer_blocks(embedded)))
        # print(raw_output.shape)
        # raw_output is batch by context_length by vocab_size

        return raw_output


# from data import train_data, test_data, encode, decode
# from model import GPT, device
# import torch
# from torch import nn

vocab_size = 104
context_length = 128 #block_size
model_dim = 252
num_blocks = 6
num_heads = 6
dropout = 0.2
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads,dropout).to(device)


