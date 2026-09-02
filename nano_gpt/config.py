import torch

text = open('input.txt', 'r', encoding='utf-8').read()

chars = sorted(list(set[str](text)))
vocab_size = len(chars)

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000

eval_iters = 200
eval_interval = 500

learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

n_head = 6
n_layer = 6
n_embd = 384 

dropout = 0.2
