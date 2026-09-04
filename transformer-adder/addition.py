import jax
from jax import nn as jnn
from jax import random
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from config import *


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
    def __init__(self, n_embed, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(n_embed, 4 * n_embed, rngs=rngs),
            nnx.gelu,
            nnx.Linear(4 * n_embed, n_embed, rngs=rngs),
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
    def __init__(self, rngs: nnx.Rngs, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.position_embedding_table = nnx.Embed(block_size, n_embed, rngs=rngs)
        self.blocks = nnx.Sequential(
            *[Block(n_embed, n_head, rngs=rngs) for _ in range(n_layer)]
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
            token_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, jnp.clip(targets, 0)
            )
            valid = targets != -1
            loss = (token_loss * valid).sum() / jnp.maximum(valid.sum(), 1)

        return logits, loss

    def genereate(self, idx, max_new_tokens, key):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            key, subkey = random.split(key)
            idx_next = jax.random.categorical(subkey, logits)[:, None]
            idx = jnp.concatenate((idx, idx_next), axis=1)

        return idx
