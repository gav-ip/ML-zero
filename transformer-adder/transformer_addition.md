---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: ML_ZERO
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gav-ip/ML-zero/blob/main/transformer_addition.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="sshRVJNNgaHL"
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import random as py_random
```

```python id="z93HjaSqRbOK"
USE_GPU = False
device = jax.devices("cpu")[0] if not USE_GPU else jax.devices()[0]
n_examples = 100000
block_size = 4
batch_size = 4
eval_iters = 200
n_embed = 32
n_heads = 4
n_blocks = 4

print(device)
```

```python colab={"base_uri": "https://localhost:8080/"} id="DrMJuaapgd7K" outputId="0de972e8-7c64-445e-a631-d9994236fb02"
vocab = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'+', 11:'=', 12:' '}
len(vocab)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eoTixx7rXhD2" outputId="d8a98a13-569c-43a0-aa03-68033cd719b6"
stoi = {vocab[i]:i for i, ch in enumerate(vocab)}
itos = {i:vocab[i] for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[int(i)] for i in l])

print(encode("8+2 =10 "))
print(decode(encode("8+7 =15 ")))
```

```python id="IYtxv3dGlC3F"
def build_key(current_key, min_val, max_val):
  # Split the key to generate 'a' and 'b', and get a new key for the next iteration
  subkey_a, subkey_b, next_key = random.split(current_key, 3)
  # Generate a single integer value using .item()
  a = random.randint(subkey_a, shape=(), minval=min_val, maxval=max_val).item()
  b = random.randint(subkey_b, shape=(), minval=min_val, maxval=max_val).item()
  return a, b, next_key
```

```python id="3OdQtfTplGkW"
def cateogrize_sum(current_key, split, min_sum_val, max_sum_val):
  target_digit_count = int(n_examples * split)

  collected_examples = []

  while (len(collected_examples) < target_digit_count):
    # Call build_key to get a single (a, b) pair and an updated key
    a, b, current_key = build_key(current_key, min_sum_val, max_sum_val) # Always generate a and b between 0-999

    s = a + b # s is now a scalar integer

    # Check if the sum 's' falls into the desired range (one-digit, two-digit, or three-digit)
    # and ensure sum s is not too large (max 3 digits for 0-999)
    if min_sum_val <= s <= max_sum_val:
      c_reversed = str(s)[::-1] # Now 's' is scalar, str(s) works as expected
      key_str = f"{a:>3}+{b:>3}={c_reversed:<4}"
      collected_examples.append(key_str)

  return collected_examples, current_key # Return the list of strings and the updated key
```

```python id="ctwukgQblz43"
def build_data():
  initial_key = random.PRNGKey(0) # Initialize JAX random key once

  one_digit_sums = []
  two_digit_sums = []
  three_digit_sums = [] # Renamed from 'other_digit_sums' for clarity

  # Pass and update the JAX random key correctly for each categorization call
  # The min_sum_val and max_sum_val here define the range for the *sum's digits*
  one_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 0, 9)
  two_digit_sums, initial_key = cateogrize_sum(initial_key, 0.25, 10, 99)
  three_digit_sums, initial_key = cateogrize_sum(initial_key, 0.5, 100, 999)

  # Combine all generated examples
  all_examples = one_digit_sums + two_digit_sums + three_digit_sums

  # Shuffle the combined list to mix the categories randomly
  py_random.shuffle(all_examples)

  text = all_examples # 'text' is now a list of formatted strings

  return text

# Call build_data and unpack the results
text = build_data()
```

```python colab={"base_uri": "https://localhost:8080/"} id="x4ek4wHbRn0S" outputId="3d924f2c-7c87-4cc1-947e-46e97d07200f"
encoded_text = [encode(t) for t in text]
# Convert the list of encoded lists into a JAX 2D array
data = jnp.array(encoded_text, device=device)

print(f"Data shape: {data.shape}")
data
```

```python
print(decode(data[1]))
```

```python
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
```

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = jnp.randint(0, len(data) - block_size, (batch_size,))
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

```
