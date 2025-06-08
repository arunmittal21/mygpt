import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        self.positions = torch.arange(context_length)
        self.position_embedding =  nn.Embedding(vocab_size, model_dim)
        self.transformer = nn.Sequential()
        for i in range(num_blocks):
            self.transformer.append(GPT.TransformerBlock(model_dim, num_heads))

        self.finalnorm = nn.LayerNorm(model_dim)
        self.projection = nn.Linear(model_dim,vocab_size)
        self.final = nn.Softmax(dim=2)


    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)
        # Round answer to 4 decimal places
        input = self.word_embedding(context) + self.position_embedding(self.positions)
        s2 = self.transformer(input)
        output_raw = self.projection(self.finalnorm(s2))
        print(output_raw)
        prob = self.final(output_raw)
        return torch.round(prob,decimals=4)

    class TransformerBlock(nn.Module):
        
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.norm1=nn.LayerNorm(model_dim)
            self.norm2=nn.LayerNorm(model_dim)
            self.attention = TransformerBlock.MultiHeadedSelfAttention(model_dim, num_heads)
            self.ff = TransformerBlock.VanillaNeuralNetwork(model_dim)
    
        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Round answer to 4 decimal places
            torch.manual_seed(0)
            s1 = self.norm1(embedded)
            s2 = self.attention(s1)
            s3 = s2+embedded
            s4 = self.norm2(s3)
            s5 = self.ff(s4)
            s6 = s5+s3
            return s6
    
        class MultiHeadedSelfAttention(nn.Module):
    
            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)
    
                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)
    
                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)
    
                    return scores @ v
                
    
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
    
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return concatenated
        
        class VanillaNeuralNetwork(nn.Module):
    
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))


