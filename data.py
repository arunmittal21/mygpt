import pandas as pd
import torch


def get_data(file="training_data/dad_jokes.csv", sep=","):
    df = pd.read_csv(file, sep=sep)
    column_text = df.iloc[:, -1].astype(str).str.cat(sep=" ")

    # df = pd.read_csv('training_data/movies-quotes.txt', sep="~")
    # df_shuffled = df.sample(frac=1, random_state=42)  # Set random_state for reproducibility
    # column_text = df_shuffled.iloc[:,2].astype(str).str.cat(sep=' ')
    print(column_text[:20])

    # print(column_text)

    chars = sorted(list(set(column_text)))
    vocab_size = len(chars)
    print(vocab_size)
    print(vocab_size)

    str_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_str = {i: ch for i, ch in enumerate(chars)}

    # encode = lambda s: [str_to_int[c] for c in s]
    # decode = lambda l: ''.join([int_to_str[i] for i in l])
    encode, decode = get_encoder_decoder(str_to_int, int_to_str)

    # Train and test splits
    data = encode(column_text)  # torch.tensor(encode(column_text), dtype=torch.long)

    n = int(0.9 * len(data))  # first 90%
    train_data = data[:n]
    test_data = data[n:]
    return vocab_size, train_data, test_data, str_to_int, int_to_str


def get_encoder_decoder(str_to_int, int_to_str):

    encode = lambda s: torch.tensor([str_to_int[c] for c in s], dtype=torch.long)
    decode = lambda l: int_to_str[l]

    return encode, decode
