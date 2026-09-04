import jax

USE_GPU = False
device = jax.devices("cpu")[0] if not USE_GPU else jax.devices()[0]
print(device)

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

vocab = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: '+', 11: '=', 12: ' ',
}
vocab_size = len(vocab)
