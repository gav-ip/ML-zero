import jax
from jax import nn as jnn
from jax import random
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import random as py_random
import optax
import pickle

USE_GPU = False
device = jax.devices("cpu")[0] if not USE_GPU else jax.devices()[0]
n_examples = 100000
block_size = 11
batch_size = 256
eval_iters = 200
max_iters = 5000
eval_interval = 500
n_embed = 32
n_heads = 4
n_layer = 6
dropout = 0.2
learning_rate = 3e-4

print(device)

vocab = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+', 11: '=', 12: ' '}
vocab_size = len(vocab)

stoi = {vocab[i]: i for i, ch in enumerate(vocab)}
itos = {i: vocab[i] for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[int(i)] for i in l])

print(encode("8+2 =10 "))
print(decode(encode("8+7 =15 ")))


def build_key(current_key, min_val, max_val):
    # Split the key to generate 'a' and 'b', and get a new key for the next iteration
    subkey_a, subkey_b, next_key = random.split(current_key, 3)
    # Generate a single integer value using .item()
    a = random.randint(subkey_a, shape=(), minval=min_val, maxval=max_val).item()
    b = random.randint(subkey_b, shape=(), minval=min_val, maxval=max_val).item()
    return a, b, next_key


def cateogrize_sum(current_key, split, min_sum_val, max_sum_val):
    target_digit_count = int(n_examples * split)

    collected_examples = []

    while len(collected_examples) < target_digit_count:

        a, b, current_key = build_key(current_key, min_sum_val, max_sum_val)  # Always generate a and b between 0-999

        s = a + b  # s is now a scalar integer

        # Check if the sum 's' falls into the desired range (one-digit, two-digit, or three-digit)
        # and ensure sum s is not too large (max 3 digits for 0-999)
        if min_sum_val <= s <= max_sum_val:
            c_reversed = str(s)[::-1]  # Now 's' is scalar, str(s) works as expected
            key_str = f"{a:>3}+{b:>3}={c_reversed:<4}"
            collected_examples.append(key_str)

    return collected_examples, current_key  # Return the list of strings and the updated key


def build_data():
    initial_key = random.PRNGKey(0)  # Initialize JAX random key once

    one_digit_sums = []
    two_digit_sums = []
    three_digit_sums = []  # Renamed from 'other_digit_sums' for clarity

    # Pass and update the JAX random key correctly for each categorization call
    # The min_sum_val and max_sum_val here define the range for the *sum's digits*
    one_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 0, 9)
    two_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 10, 99)
    three_digit_sums, initial_key = cateogrize_sum(initial_key, 0.5, 100, 999)

    all_examples = one_digit_sums + two_digit_sums + three_digit_sums

    py_random.shuffle(all_examples)

    text = all_examples

    return text


text = build_data()

encoded_text = [encode(t) for t in text]
# Convert the list of encoded lists into a JAX 2D array
data = jnp.array(encoded_text, device=device)

print(f"Data shape: {data.shape}")
print(data)

print(decode(data[2]))
print(data[2])
print(len(data[2]))

n = int(0.9 * len(data))  # first 90% will be train, rest val
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


rngs = nnx.Rngs(0)


class MultiHeadAttention(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.key = nnx.Linear(n_embed, n_embed, use_bias=False, rngs=rngs)
        self.query = nnx.Linear(n_embed, n_embed, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(n_embed, n_embed, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(n_embed, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        head_dim = C // n_heads
        k = self.key(x).reshape(B, T, n_heads, head_dim)
        q = self.query(x).reshape(B, T, n_heads, head_dim)
        v = self.value(x).reshape(B, T, n_heads, head_dim)

        out = jnn.dot_product_attention(q, k, v, mask=None, is_causal=True)
        out = out.reshape(B, T, C)

        return self.dropout(self.proj(out))


class FeedForward(nnx.Module):
    def __init__(self, x: n_embed, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(x, 4 * x, rngs=rngs),
            nnx.gelu,
            nnx.Linear(4 * x, x, rngs=rngs),
            nnx.Dropout(dropout, rngs=rngs),
        )

    def __call__(self, x):
        return self.net(x)


class Block(nnx.Module):
    def __init__(self, n_embd, n_head, *, rngs: nnx.Rngs):
        self.sa = MultiHeadAttention(rngs=rngs)
        self.ff = FeedForward(n_embd, rngs=rngs)
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class AdditionModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, block_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.position_embedding_table = nnx.Embed(block_size, n_embed, rngs=rngs)
        self.blocks = nnx.Sequential(
            *[Block(n_embed, n_heads, rngs=rngs) for _ in range(n_layer)]
        )
        self.ln_f = nnx.LayerNorm(n_embed, rngs=rngs)
        self.lm_head = nnx.Linear(n_embed, vocab_size, rngs=rngs)

    def __call__(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(jnp.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return logits, loss

    def genereate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            key, subkey = random.split(key)
            idx_next = jax.random.categorical(subkey, logits)

            idx = jax.concatenate((idx, idx_next), axis=1)

        return idx


model = AdditionModel(rngs=nnx.Rngs(0), block_size=block_size, n_embed=n_embed, n_head=n_heads, n_layer=n_layer, dropout=dropout)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param)

key = random.PRNGKey(0)


# call jax.lax.stop_gradient for no_grad
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
    return out, key


# training loop
key = random.PRNGKey(1)
for steps in range(max_iters):
    key, eval_key, batch_key = random.split(key, 3)
    if steps % eval_interval == 0:
        losses = estimate_loss(eval_key)
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # get sample batch from training set
    xb, yb = get_batch(key, 'train')

    logits, loss = model(xb, yb)
    loss, grads = nnx.value_and_grad(loss)(model, xb, yb)
    _ = optimizer.update(model, grads)

checkpoint = {
    'model_state_dict': nnx.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embed,
        'n_head': n_heads,
        'n_layer': n_layer,
        'dropout': dropout,
    }
}
with open('nano_gpt.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
