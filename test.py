# from model_init import MY_MODEL as model,context_length, device
from model import GPT, device
# from data import decode, int_to_str
from data import get_encoder_decoder
import torch
import os

out_dir='checkpoint'

# checkpoint = {
#     'model': model.state_dict(),
#     'vocab_size': model.token_embedding.num_embeddings,
#     'context_length': context_length,
#     'model_dim': model.token_embedding.embedding_dim,
#     'num_blocks': len(model.blocks),
#     'num_heads': model.blocks[0].attention.num_heads,
#     'dropout': 0.2
# }

def load_model(weights='ckpt.pt'):
    checkpoint_path = os.path.join(out_dir, weights)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        context_length = int(checkpoint['context_length'])
        model = GPT(
            vocab_size=checkpoint['vocab_size'],
            context_length=checkpoint['context_length'],
            model_dim=checkpoint['model_dim'],
            num_blocks=checkpoint['num_blocks'],
            num_heads=checkpoint['num_heads'],
            dropout=checkpoint['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("Model loaded successfully.")
        encode, decode = get_encoder_decoder(checkpoint['str_to_int'],checkpoint['int_to_str'])
        return model,context_length
    else:
        print(f"No checkpoint found at {checkpoint_path}.")
        return None, None



from typing import Generator

def generate(model, new_chars: int, context, context_length: int, int_to_str: dict, temperature = 1.0   ) -> Generator[str, None, None]:
    res = []
    for i in range(new_chars):
        if len(context.T) > context_length:
            context = context[:, -context_length:]
        prediction = model(context) # B, T, Vocab_Size
        last_time_step = prediction[:, -1, :] # B, Vocab_Size
        probabilities = torch.nn.functional.softmax(last_time_step/temperature, dim = -1)
        # values, max_index = torch.max(probabilities,dim=1)
        # next_char = max_index.unsqueeze(1)
        next_char = torch.multinomial(probabilities, 1)
        # print (next_char)
        context = torch.cat((context, next_char), dim = -1)
        next_char_decoded = int_to_str.get(next_char.item(),' ')
        yield next_char_decoded
    # return ''.join(res)

def test(file, generate_len = 300):
    # model = load_model('dadjokes-2.pt')
    model, context_length = load_model(file)
    # generate_len = 300  # block_size
    
    if model is not None and context_length is not None:
        context = torch.zeros(1, 1, dtype=torch.int64).to(device)
        print(context)
    
        for char in generate(model, new_chars=generate_len, context=context,
                             context_length=context_length, int_to_str=int_to_str,
                             temperature=0.5):
            print(char, end='')  # print each character without newline
    
        print(' ')
        print('------------------')
    else:
        print("Failed to load model or context_length.")