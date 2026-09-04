import pickle

import flax.nnx as nnx
import jax.numpy as jnp
from jax import random

from addition import AdditionModel
from config import device

with open('transformer_addition.pkl', 'rb') as f:
    ckpt = pickle.load(f)

cfg = ckpt['config']
stoi, itos = ckpt['stoi'], ckpt['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda a: ''.join([itos[int(i)] for i in a])

m = AdditionModel(rngs=nnx.Rngs(0), **cfg)
nnx.update(m, ckpt['model_state'])
m.eval()

prompt = "  8+  7="
idx = jnp.array([encode(prompt)], device=device)
out = m.genereate(idx, max_new_tokens=4, key=random.PRNGKey(0))
print(decode(out[0]))
