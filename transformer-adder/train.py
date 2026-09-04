import pickle
import random as py_random

import flax.nnx as nnx
import jax.numpy as jnp
import optax
from jax import random

from addition import AdditionModel
from config import *

stoi = {vocab[i]: i for i, ch in enumerate(vocab)}
itos = {i: vocab[i] for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[int(i)] for i in l])


def build_key(current_key, min_val, max_val):
    subkey_a, subkey_b, next_key = random.split(current_key, 3)
    a = random.randint(subkey_a, shape=(), minval=min_val, maxval=max_val).item()
    b = random.randint(subkey_b, shape=(), minval=min_val, maxval=max_val).item()
    return a, b, next_key


def cateogrize_sum(current_key, split, min_sum_val, max_sum_val):
    target_digit_count = int(n_examples * split)
    collected_examples = []

    while len(collected_examples) < target_digit_count:
        a, b, current_key = build_key(current_key, min_sum_val, max_sum_val)
        s = a + b
        if min_sum_val <= s <= max_sum_val:
            c_reversed = str(s)[::-1]
            key_str = f"{a:>3}+{b:>3}={c_reversed:<4}"
            collected_examples.append(key_str)

    return collected_examples, current_key


def build_data():
    initial_key = random.PRNGKey(0)
    one_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 0, 9)
    two_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 10, 99)
    three_digit_sums, initial_key = cateogrize_sum(initial_key, 0.5, 100, 999)
    all_examples = one_digit_sums + two_digit_sums + three_digit_sums
    py_random.shuffle(all_examples)
    return all_examples


text = build_data()
encoded_text = [encode(t) for t in text]
data = jnp.array(encoded_text, device=device)
print(f"Data shape: {data.shape}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(key, split):
    data = train_data if split == 'train' else val_data
    ix = random.randint(key, (batch_size,), 0, len(data))
    seq = data[ix]
    x = seq[:, :-1]
    y = seq[:, 1:]
    y = y.at[:, :7].set(-1)
    return x, y


def estimate_loss(key):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            key, subkey = random.split(key)
            X, Y = get_batch(subkey, split)
            _, loss = model(X, Y)
            losses = losses.at[k].set(loss)
        out[split] = losses.mean()
    model.train()
    return out


model = AdditionModel(
    rngs=nnx.Rngs(0),
    vocab_size=vocab_size,
    block_size=block_size,
    n_embed=n_embed,
    n_head=n_heads,
    n_layer=n_layer,
    dropout=dropout,
)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param)


def loss_fn(model, xb, yb):
    _, loss = model(xb, yb)
    return loss


key = random.PRNGKey(1)
for steps in range(max_iters):
    key, eval_key, batch_key = random.split(key, 3)
    if steps % eval_interval == 0:
        losses = estimate_loss(eval_key)
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(batch_key, 'train')
    loss, grads = nnx.value_and_grad(loss_fn)(model, xb, yb)
    optimizer.update(model, grads)

checkpoint = {
    'model_state': nnx.state(model),
    'stoi': stoi,
    'itos': itos,
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embed': n_embed,
        'n_head': n_heads,
        'n_layer': n_layer,
        'dropout': dropout,
    },
}
with open('transformer_addition.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
print('saved checkpoint to transformer_addition.pkl')
