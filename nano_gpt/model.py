import os
import torch
from gpt import BigramLanguageModel, device  # no training side effects

ckpt = torch.load('nano_gpt.pt', map_location=device)
cfg = ckpt['config']
stoi, itos = ckpt['stoi'], ckpt['itos']
decode = lambda a: ''.join([itos[i] for i in a])

# ideally set hyperparams from cfg before building model
m = BigramLanguageModel(**cfg).to(device)  # if __init__ accepts them
m.load_state_dict(ckpt['model_state_dict'])
m.eval()

with torch.no_grad():
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.genereate(idx, max_new_tokens=500)[0].tolist()))