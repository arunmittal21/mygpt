from model import model,context_length, device
from data import decode, int_to_str
import torch


def generate(model, new_chars: int, context, context_length: int, int_to_char: dict, temperature = 1.0   ) -> str:
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
        next_char_decoded = int_to_str[next_char.item()]
        yield next_char_decoded
    # return ''.join(res)


context = torch.zeros(1, 1, dtype = torch.int64).to(device)
print(context)


for char in generate(model, new_chars=10, context=context,
                     context_length=context_length, int_to_char=int_to_char,
                     temperature=2):
    # print(char, end='')  # print each character without newline
    # print(char)  # print each character without newline
    pass
