import pandas as pd
import torch 

# df = pd.read_csv('training_data/dad_jokes.csv')
# column_text = df.iloc[:, 1].astype(str).str.cat(sep=' ')

df = pd.read_csv('training_data/movies-quotes.txt', sep="~")
df_shuffled = df.sample(frac=1, random_state=42)  # Set random_state for reproducibility
column_text = df_shuffled.iloc[:,2].astype(str).str.cat(sep=' ')
print (column_text[:20])

# print(column_text)

chars = sorted(list(set(column_text)))
vocab_size = len(chars)
print(vocab_size)
print(vocab_size)

str_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_str = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [str_to_int[c] for c in s] 
decode = lambda l: ''.join([int_to_str[i] for i in l])


# Train and test splits
data = torch.tensor(encode(column_text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% 
train_data = data[:n]
test_data = data[n:]


